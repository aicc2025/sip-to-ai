"""RTP session with precise 20ms timing using asyncio.

This module implements RTP (Real-time Transport Protocol) for audio streaming
with G.711 codec support and precise 20ms frame timing.
"""

import asyncio
import audioop
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

import structlog

logger = structlog.get_logger(__name__)


class PortBindError(Exception):
    """Raised when RTP port binding fails (port already in use)."""
    def __init__(self, port: int, original_error: Exception):
        self.port = port
        self.original_error = original_error
        super().__init__(f"Failed to bind RTP port {port}: {original_error}")


@dataclass
class RTPConfig:
    """RTP configuration for audio streaming."""

    sample_rate: int = 8000  # Hz
    frame_ms: int = 20  # milliseconds
    payload_type: int = 0  # PCMU (G.711 μ-law)

    @property
    def frame_samples(self) -> int:
        """Number of samples per frame."""
        return (self.sample_rate * self.frame_ms) // 1000

    @property
    def frame_interval(self) -> float:
        """Frame interval in seconds."""
        return self.frame_ms / 1000.0

    @property
    def pcm16_frame_size(self) -> int:
        """PCM16 frame size in bytes (16-bit = 2 bytes/sample)."""
        return self.frame_samples * 2

    @property
    def g711_frame_size(self) -> int:
        """G.711 frame size in bytes (8-bit = 1 byte/sample)."""
        return self.frame_samples


# Static RTP payload types (RFC 3551)
PT_PCMU = 0
PT_PCMA = 8


class G711Codec:
    """G.711 audio codec (PCMU/PCMA) using audioop."""

    def encode(self, pcm16: bytes, payload_type: int) -> bytes:
        """Encode PCM16 with the codec for the RTP payload type (PCMA or PCMU)."""
        if payload_type == PT_PCMA:
            return self.encode_pcma(pcm16)
        return self.encode_pcmu(pcm16)

    def decode(self, payload: bytes, payload_type: int) -> bytes:
        """Decode an RTP payload to PCM16 with the codec for its payload type."""
        if payload_type == PT_PCMA:
            return self.decode_pcma(payload)
        return self.decode_pcmu(payload)

    def encode_pcmu(self, pcm16: bytes) -> bytes:
        """Encode PCM16 to G.711 μ-law.

        Args:
            pcm16: PCM 16-bit audio data

        Returns:
            G.711 μ-law encoded data (half the size)
        """
        return audioop.lin2ulaw(pcm16, 2)

    def decode_pcmu(self, ulaw: bytes) -> bytes:
        """Decode G.711 μ-law to PCM16.

        Args:
            ulaw: G.711 μ-law encoded data

        Returns:
            PCM 16-bit audio data (double the size)
        """
        return audioop.ulaw2lin(ulaw, 2)

    def encode_pcma(self, pcm16: bytes) -> bytes:
        """Encode PCM16 to G.711 A-law."""
        return audioop.lin2alaw(pcm16, 2)

    def decode_pcma(self, alaw: bytes) -> bytes:
        """Decode G.711 A-law to PCM16."""
        return audioop.alaw2lin(alaw, 2)


class RTPPacket:
    """RTP packet structure (inspired by pyVoIP)."""

    def __init__(self, data: bytes):
        """Parse RTP packet from raw bytes.

        Args:
            data: Raw RTP packet bytes
        """
        if len(data) < 12:
            raise ValueError("RTP packet too short")

        # Parse RTP header (RFC 3550)
        self.version = (data[0] >> 6) & 0x3
        self.padding = bool(data[0] & 0x20)
        self.extension = bool(data[0] & 0x10)
        self.csrc_count = data[0] & 0xF

        self.marker = bool(data[1] & 0x80)
        self.payload_type = data[1] & 0x7F

        self.sequence = int.from_bytes(data[2:4], 'big')
        self.timestamp = int.from_bytes(data[4:8], 'big')
        self.ssrc = int.from_bytes(data[8:12], 'big')

        # Extract payload (skip header + CSRC)
        header_size = 12 + (self.csrc_count * 4)
        self.payload = data[header_size:]

    @staticmethod
    def build(
        payload: bytes,
        seq: int,
        timestamp: int,
        ssrc: int,
        pt: int = 0,
        marker: bool = False
    ) -> bytes:
        """Build RTP packet from components.

        Args:
            payload: Audio payload data
            seq: Sequence number
            timestamp: RTP timestamp
            ssrc: Synchronization source identifier
            pt: Payload type (0 = PCMU)
            marker: Marker bit

        Returns:
            Complete RTP packet bytes
        """
        packet = bytearray()

        # Byte 0: V=2, P=0, X=0, CC=0
        packet.append(0x80)

        # Byte 1: M (marker), PT (payload type)
        byte1 = pt & 0x7F
        if marker:
            byte1 |= 0x80
        packet.append(byte1)

        # Bytes 2-3: Sequence number
        packet.extend(seq.to_bytes(2, 'big'))

        # Bytes 4-7: Timestamp
        packet.extend(timestamp.to_bytes(4, 'big'))

        # Bytes 8-11: SSRC
        packet.extend(ssrc.to_bytes(4, 'big'))

        # Payload
        packet.extend(payload)

        return bytes(packet)


class RTPProtocol(asyncio.DatagramProtocol):
    """Asyncio datagram protocol for receiving RTP packets."""

    def __init__(self, session: 'RTPSession'):
        """Initialize RTP protocol.

        Args:
            session: Parent RTP session
        """
        self.session = session
        self.transport: Optional[asyncio.DatagramTransport] = None
        self._logged_first_packet = False
        self._ignored_packets = 0
        self._invalid_packets = 0
        self._rejected_source_packets = 0

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        """Called when UDP socket is ready."""
        self.transport = transport  # type: ignore
        self.session.transport = transport  # type: ignore
        logger.debug("RTP protocol connection made")

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Called when RTP packet is received (event-driven, no explicit timing needed).

        Args:
            data: Raw packet bytes
            addr: Source address
        """
        try:
            # Parse RTP packet
            rtp = RTPPacket(data)

            # Only the negotiated audio codec is decoded; other payload types
            # (telephone-event, comfort noise, stray packets) are ignored
            expected_pt = self.session.config.payload_type
            if rtp.version != 2 or rtp.payload_type != expected_pt:
                self._ignored_packets += 1
                if self._ignored_packets == 1 or self._ignored_packets % 500 == 0:
                    logger.debug(
                        "Ignoring RTP packet with unexpected version/payload type",
                        version=rtp.version,
                        payload_type=rtp.payload_type,
                        expected_payload_type=expected_pt,
                        ignored_total=self._ignored_packets,
                        source=addr,
                    )
                return

            # Symmetric RTP: send to where the caller's RTP actually comes from.
            # Packets from any other source are dropped (never reach the AI).
            if not self.session.observe_source(addr, rtp.ssrc):
                self._rejected_source_packets += 1
                self.session.rejected_source_packets = self._rejected_source_packets
                if self._rejected_source_packets == 1 or self._rejected_source_packets % 500 == 0:
                    logger.debug(
                        "Dropping RTP packet from non-latched source",
                        source=addr,
                        ssrc=rtp.ssrc,
                        latched_remote_addr=self.session.remote_addr,
                        dropped_total=self._rejected_source_packets,
                    )
                return

            # Decode G.711 to PCM16
            pcm = self.session.codec.decode(rtp.payload, rtp.payload_type)

            # Make the inbound caller-audio path visible on first packet (issue #6:
            # "caller audio is not logged"). Confirms RTP is actually arriving.
            if not self._logged_first_packet:
                self._logged_first_packet = True
                logger.info(
                    "First inbound RTP packet received from caller",
                    payload_type=rtp.payload_type,
                    payload_bytes=len(rtp.payload),
                    decoded_pcm16_bytes=len(pcm),
                    source=addr,
                )

            # Put into receive queue (non-blocking)
            try:
                self.session.rx_queue.put_nowait(pcm)
            except asyncio.QueueFull:
                # Queue full - drop oldest frame to prevent latency buildup
                try:
                    self.session.rx_queue.get_nowait()
                    self.session.rx_queue.put_nowait(pcm)
                except asyncio.QueueEmpty:
                    pass

        except Exception as e:
            # Garbage/truncated datagrams: count them, log only periodically
            self._invalid_packets += 1
            self.session.invalid_packets = self._invalid_packets
            if self._invalid_packets == 1 or self._invalid_packets % 100 == 0:
                logger.debug(
                    "Ignoring invalid RTP datagram",
                    error=str(e),
                    source=addr,
                    size=len(data),
                    invalid_total=self._invalid_packets,
                )

    def error_received(self, exc: Exception) -> None:
        """Called when socket error occurs."""
        logger.error("RTP protocol error", error=str(exc))


class RTPSession:
    """RTP session with precise 20ms timing using asyncio.TaskGroup."""

    # Consecutive valid packets (same source address and SSRC) needed to latch
    # to a source that does not match the SDP address (NAT)
    LATCH_PACKETS = 2
    # A latched source must be silent this long before another source may take over
    LATCH_MOVE_SILENCE = 0.5
    # Steady packets (same source and SSRC) a new source needs to take over the
    # latch, or to latch when it is a stale pre-re-INVITE source
    LATCH_MOVE_PACKETS = 10
    # Maximum gap between packets of a candidate source to count as steady
    LATCH_STEADY_GAP = 0.1

    def __init__(
        self,
        local_port: int,
        remote_addr: tuple[str, int],
        config: Optional[RTPConfig] = None
    ):
        """Initialize RTP session.

        Args:
            local_port: Local UDP port for RTP
            remote_addr: Remote (IP, port) for sending RTP
            config: RTP configuration
        """
        self.local_port = local_port
        # remote_addr is where RTP is sent; sdp_remote_addr is what SDP advertised
        self.remote_addr = remote_addr
        self.sdp_remote_addr = remote_addr
        self.config = config or RTPConfig()
        self.codec = G711Codec()

        # Audio queues (1 second buffer = 50 frames)
        self.rx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self.tx_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)

        # RTP state (randomized per RFC 3550)
        self.sequence_num = random.randint(0, 65535)
        self.timestamp = random.randint(0, 0xFFFFFFFF)
        self.ssrc = random.randint(0, 0xFFFFFFFF)

        self.transport: Optional[asyncio.DatagramTransport] = None
        self._transport_lock = asyncio.Lock()
        self._running = False

        # Symmetric RTP (latching) state
        self._latched = False
        self._latched_ssrc: Optional[int] = None
        self._latched_last_rx = 0.0
        self._latch_candidate: Optional[tuple[tuple[str, int], int]] = None  # (addr, ssrc)
        self._latch_count = 0
        self._latch_candidate_last_rx = 0.0
        # Sources of the previous SDP address (before a re-INVITE changed it)
        self._stale_sources: set[tuple[str, int]] = set()

        # Outbound media enabled (False while on hold: recvonly/inactive answer
        # or c=0.0.0.0)
        self._send_enabled = True

        # Inbound packet counters (updated by RTPProtocol)
        self.invalid_packets = 0
        self.rejected_source_packets = 0

        # Statistics
        self._frames_sent = 0
        self._frames_received = 0
        self._silence_frames = 0

    def update_port(self, new_port: int) -> None:
        """Update local RTP port (before start).

        Args:
            new_port: New RTP port number
        """
        if self._running:
            raise RuntimeError("Cannot update port while session is running")
        self.local_port = new_port

    @property
    def latched(self) -> bool:
        """True once the send address has been latched to the observed source."""
        return self._latched

    @property
    def send_enabled(self) -> bool:
        """True if outbound RTP is sent (False while the call is on hold)."""
        return self._send_enabled

    def set_send_enabled(self, enabled: bool) -> None:
        """Enable or disable outbound RTP (hold/resume).

        While disabled, queued downlink audio is discarded at the normal 20ms
        pace and no packets are sent.

        Args:
            enabled: True to send RTP, False to stop sending
        """
        if enabled == self._send_enabled:
            return
        self._send_enabled = enabled
        logger.info(
            "Outbound RTP resumed" if enabled else "Outbound RTP paused (hold)",
            remote_addr=self.remote_addr,
        )

    def observe_source(self, addr: tuple[str, int], ssrc: int, now: Optional[float] = None) -> bool:
        """Track the source of the caller's RTP (symmetric RTP) and filter sources.

        Called for each valid inbound packet (expected payload type). Rules:

        - A packet from the SDP address latches immediately (preferred source).
        - A packet from another address latches after LATCH_PACKETS consecutive
          packets with the same SSRC when nothing is latched yet (callers behind
          NAT/proxy/TUN whose SDP address is not where RTP comes from).
        - Sources of the previous SDP address (before a re-INVITE) need
          LATCH_MOVE_PACKETS steady packets, so in-flight packets from the old
          port never win over the new SDP address.
        - Once latched, another source can take over only after the latched
          source has been silent for LATCH_MOVE_SILENCE and the new source sent
          LATCH_MOVE_PACKETS steady packets with one SSRC. A packet from the SDP
          address always takes over.

        Args:
            addr: Source (IP, port) of the packet
            ssrc: RTP SSRC of the packet
            now: Monotonic time (defaults to time.monotonic())

        Returns:
            True if the packet comes from the latched source and must be used,
            False if it must be dropped
        """
        if now is None:
            now = time.monotonic()
        source = (addr[0], int(addr[1]))

        if self._latched and source == tuple(self.remote_addr):
            self._latched_last_rx = now
            self._latched_ssrc = ssrc
            return True

        if source == tuple(self.sdp_remote_addr):
            self._latch(source, ssrc, now, reason="sdp address")
            return True

        # Candidate tracking: same source and SSRC with no large gap
        if (
            self._latch_candidate == (source, ssrc)
            and now - self._latch_candidate_last_rx <= self.LATCH_STEADY_GAP
        ):
            self._latch_count += 1
        else:
            self._latch_candidate = (source, ssrc)
            self._latch_count = 1
        self._latch_candidate_last_rx = now

        if self._latched:
            if now - self._latched_last_rx < self.LATCH_MOVE_SILENCE:
                return False
            required = self.LATCH_MOVE_PACKETS
        elif source in self._stale_sources:
            required = self.LATCH_MOVE_PACKETS
        else:
            required = self.LATCH_PACKETS

        if self._latch_count < required:
            return False

        self._latch(source, ssrc, now, reason="observed source")
        return True

    def _latch(self, source: tuple[str, int], ssrc: int, now: float, reason: str) -> None:
        """Latch the send address to source."""
        previous = tuple(self.remote_addr) if self._latched else None
        self._latched = True
        self._latched_ssrc = ssrc
        self._latched_last_rx = now
        self._latch_candidate = None
        self._latch_count = 0
        self._stale_sources.clear()

        if source != tuple(self.remote_addr):
            logger.info(
                "Symmetric RTP: sending to observed caller RTP source",
                sdp_remote_addr=self.sdp_remote_addr,
                previous_remote_addr=previous,
                latched_remote_addr=source,
                ssrc=ssrc,
                reason=reason,
            )
            self.remote_addr = source

    def update_remote(self, remote_addr: tuple[str, int], payload_type: int) -> None:
        """Apply a changed remote SDP (re-INVITE).

        A new SDP address resets latching so it can re-latch to the new source;
        the previous SDP and latched addresses become stale sources that cannot
        immediately win the latch.

        Args:
            remote_addr: Remote (IP, port) from the new SDP
            payload_type: Negotiated RTP payload type
        """
        if tuple(remote_addr) != tuple(self.sdp_remote_addr):
            logger.info(
                "Remote RTP address changed",
                old_sdp_remote_addr=self.sdp_remote_addr,
                new_sdp_remote_addr=remote_addr,
            )
            self._stale_sources = {
                (self.sdp_remote_addr[0], self.sdp_remote_addr[1]),
                (self.remote_addr[0], int(self.remote_addr[1])),
            }
            self._stale_sources.discard(tuple(remote_addr))
            self.sdp_remote_addr = remote_addr
            self.remote_addr = remote_addr
            self._latched = False
            self._latched_ssrc = None
            self._latch_candidate = None
            self._latch_count = 0

        if payload_type != self.config.payload_type:
            logger.info(
                "RTP payload type changed",
                old_payload_type=self.config.payload_type,
                new_payload_type=payload_type,
            )
            self.config.payload_type = payload_type

    def clear_tx_queue(self) -> int:
        """Drop queued outbound audio frames (barge-in).

        Returns:
            Number of frames dropped
        """
        dropped = 0
        while True:
            try:
                self.tx_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            dropped += 1
        return dropped

    async def start(self) -> None:
        """Start RTP session (create UDP endpoint).

        Raises:
            PortBindError: If port is already in use
        """
        loop = asyncio.get_running_loop()

        try:
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: RTPProtocol(self),
                local_addr=('0.0.0.0', self.local_port)
            )
        except OSError as e:
            # Port already in use or other bind error
            logger.error(
                "Failed to bind RTP port",
                port=self.local_port,
                error=str(e)
            )
            raise PortBindError(self.local_port, e) from e

        # Set transport with lock protection
        async with self._transport_lock:
            self.transport = transport

        self._running = True
        logger.info(
            "RTP session started",
            local_port=self.local_port,
            remote_addr=self.remote_addr,
            frame_ms=self.config.frame_ms
        )

    async def run(self) -> None:
        """Run RTP session with TaskGroup."""
        await self.start()

        try:
            async with asyncio.TaskGroup() as tg:
                # Send loop with precise 20ms timing
                tg.create_task(
                    self._send_loop(),
                    name=f"rtp-send-{self.local_port}"
                )
                # Receive is event-driven via DatagramProtocol
                logger.info("RTP TaskGroup started")

        except* asyncio.CancelledError:
            # Normal cancellation during shutdown
            logger.debug("RTP tasks cancelled (normal shutdown)")

        except* Exception as eg:
            # Unexpected exceptions
            logger.error(
                "RTP TaskGroup exceptions",
                count=len(eg.exceptions),
                local_port=self.local_port
            )
            for exc in eg.exceptions:
                logger.error(
                    f"Exception: {type(exc).__name__}: {exc}",
                    exc_info=exc
                )

        finally:
            await self.stop()

    async def _send_loop(self) -> None:
        """Precise 20ms timing send loop (critical implementation).

        This loop maintains exact timing by:
        1. Using absolute time (loop.time()) to avoid drift
        2. Compensating for processing time
        3. Sending silence when no data available
        """
        interval = self.config.frame_interval  # 0.02 seconds
        frame_samples = self.config.frame_samples  # 160 samples

        # Use absolute time to avoid cumulative error
        loop = asyncio.get_running_loop()
        next_send_time = loop.time()

        # Pre-allocate silence frame
        silence_pcm16 = b'\x00' * self.config.pcm16_frame_size

        logger.info(
            "RTP send loop started",
            interval_ms=interval * 1000,
            frame_samples=frame_samples
        )

        while self._running:
            # 1. Get audio data from queue (with timeout)
            try:
                pcm_data = await asyncio.wait_for(
                    self.tx_queue.get(),
                    timeout=interval * 0.5  # Wait max 10ms
                )
            except asyncio.TimeoutError:
                # No data - send silence
                pcm_data = silence_pcm16
                self._silence_frames += 1

            if not self._send_enabled:
                # On hold: send nothing, but keep the pace and the RTP clock
                self.timestamp = (self.timestamp + frame_samples) % 0x100000000
                next_send_time += interval
                await asyncio.sleep(max(0, next_send_time - loop.time()))
                continue

            # 2. Encode PCM16 to G.711 (negotiated PCMU or PCMA)
            ulaw_data = self.codec.encode(pcm_data, self.config.payload_type)

            # 3. Build and send RTP packet
            packet = RTPPacket.build(
                payload=ulaw_data,
                seq=self.sequence_num,
                timestamp=self.timestamp,
                ssrc=self.ssrc,
                pt=self.config.payload_type
            )

            # Send with lock protection (thread-safe)
            async with self._transport_lock:
                if self.transport:
                    try:
                        self.transport.sendto(packet, self.remote_addr)
                        self._frames_sent += 1
                        # Confirm the outbound caller-audio path on first packet
                        # (issue #6: "caller does not hear any audio").
                        if self._frames_sent == 1:
                            logger.info(
                                "First outbound RTP packet sent to caller",
                                payload_bytes=len(ulaw_data),
                                remote_addr=self.remote_addr,
                            )
                    except (AttributeError, OSError) as e:
                        # Transport closed during send - stop gracefully
                        logger.debug(
                            "Transport closed during send",
                            error=str(e),
                            local_port=self.local_port
                        )
                        break

            # 4. Update RTP state
            self.sequence_num = (self.sequence_num + 1) % 65536
            self.timestamp = (self.timestamp + frame_samples) % 0x100000000

            # 5. Precise sleep with drift correction
            next_send_time += interval
            now = loop.time()
            sleep_time = max(0, next_send_time - now)

            # Detect and correct large time drift
            if sleep_time > interval * 2:
                logger.warning(
                    "RTP timing drift detected",
                    sleep_time=sleep_time,
                    interval=interval
                )
                next_send_time = now + interval
                sleep_time = interval

            await asyncio.sleep(sleep_time)

            # Periodic logging
            if self._frames_sent % 500 == 0:
                logger.debug(
                    "RTP send stats",
                    frames_sent=self._frames_sent,
                    silence_frames=self._silence_frames,
                    silence_ratio=f"{100 * self._silence_frames / max(1, self._frames_sent):.1f}%"
                )

    async def receive_audio(self) -> AsyncIterator[bytes]:
        """Async iterator for receiving PCM16 audio frames.

        Yields:
            PCM16 audio frames (320 bytes @ 8kHz, 20ms)

        Note:
            This iterator runs independently of _running flag.
            It will block on queue.get() until data is available or queue is closed.
        """
        while True:
            try:
                pcm_data = await self.rx_queue.get()
                self._frames_received += 1
                yield pcm_data
            except asyncio.CancelledError:
                # Task was cancelled - propagate to caller
                logger.debug("RTP receive_audio iterator cancelled")
                raise
            except Exception as e:
                logger.error("RTP receive_audio error", error=str(e))
                break

    async def receive_frame(self, timeout: float) -> Optional[bytes]:
        """Receive one PCM16 audio frame, waiting at most timeout seconds.

        Args:
            timeout: Maximum wait in seconds

        Returns:
            PCM16 frame (320 bytes @ 8kHz), or None if no RTP arrived in time
        """
        try:
            pcm_data = await asyncio.wait_for(self.rx_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        self._frames_received += 1
        return pcm_data

    async def send_audio(self, pcm_data: bytes) -> None:
        """Send PCM16 audio frame to RTP.

        Args:
            pcm_data: PCM16 audio frame (320 bytes @ 8kHz, 20ms)
        """
        await self.tx_queue.put(pcm_data)

    async def stop(self) -> None:
        """Stop RTP session (thread-safe, idempotent)."""
        self._running = False

        # Close and clear transport with lock protection
        async with self._transport_lock:
            if not self.transport:
                return
            try:
                self.transport.close()
            except Exception as e:
                logger.debug("Error closing transport", error=str(e))
            finally:
                self.transport = None

        logger.info(
            "RTP session stopped",
            frames_sent=self._frames_sent,
            frames_received=self._frames_received,
            invalid_packets=self.invalid_packets,
            rejected_source_packets=self.rejected_source_packets
        )
