# pi-protocol

Python port of `packages/protocol/src`: protocol version 8, strict envelope
validation, CBOR encoding, and incremental byte-stream framing.

Wire dictionaries retain the TypeScript field names (`serverId`, `sessionId`,
`attachmentId`, `subscriptionId`). Python functions and option fields use
snake_case. CBOR and complete frame encoders return `bytes`; decoders accept
`bytes`, `bytearray`, and contiguous byte `memoryview` inputs.

```python
from pi_protocol import (
    PROTOCOL_VERSION,
    FrameDecoderOptions,
    ServerMessageDecoder,
    encode_client_message,
)

hello = encode_client_message({"type": "hello", "version": PROTOCOL_VERSION})
decoder = ServerMessageDecoder(FrameDecoderOptions(max_frame_length=1024 * 1024))
```

Pass incoming byte chunks to `decoder.push(chunk)` and process the returned
messages. Call `decoder.end()` when the byte stream ends; incomplete frames
raise `ProtocolValidationError` and a failed decoder remains unusable.

The outer envelopes reject unknown fields. Their opaque payloads must be
strict JSON; Chord owns service-call, catalogue, snapshot, and update grammar.
The default limits match TypeScript: 16 MiB per frame/CBOR item, 1,000,000
elements or map entries, and 64 nested CBOR item levels.

CBOR map keys preserve JavaScript object-key order, including numeric index
keys. This matches the TypeScript encoder's ordering; RFC canonical sorting
is not applied. Invalid option limits raise Python `ValueError` with the
original range-error message.

This package belongs to the `py/` uv workspace and depends on `pi-chord`.
Implementation is in progress across the Python port; validation and testing
are deferred to the final unified testing phase.
