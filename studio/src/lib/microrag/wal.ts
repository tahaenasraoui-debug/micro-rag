/** In-memory append-only WAL with the same IEEE CRC32 as micro-rag `storage.rs`. */

import type { DocumentRecord } from "./index";

export function crc32Ieee(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc ^= bytes[i];
    for (let bit = 0; bit < 8; bit++) {
      const mask = -(crc & 1) >>> 0;
      crc = (crc >>> 1) ^ (0xedb88320 & mask);
    }
    crc >>>= 0;
  }
  return (~crc) >>> 0;
}

export type WalFrame = {
  n: number;
  crc: number;
  payload: string;
  id: string;
  text: string;
  domain: string;
  bad: boolean;
};

function payloadFor(record: DocumentRecord): string {
  const metaKeys = Object.keys(record.metadata).sort();
  const metadata: Record<string, string> = {};
  for (const key of metaKeys) metadata[key] = record.metadata[key];
  return JSON.stringify({
    op: "upsert",
    id: record.id,
    text: record.text,
    metadata,
    vector: record.vector.map((n) => Math.round(n * 1e6) / 1e6),
  });
}

export function frameFromRecord(n: number, record: DocumentRecord): WalFrame {
  const payload = payloadFor(record);
  const crc = crc32Ieee(new TextEncoder().encode(payload));
  return {
    n,
    crc,
    payload,
    id: record.id,
    text: record.text,
    domain: record.metadata.domain ?? "note",
    bad: false,
  };
}

export function verifyFrame(frame: WalFrame): { ok: boolean; actual: number } {
  const actual = crc32Ieee(new TextEncoder().encode(frame.payload));
  return { ok: actual === frame.crc && !frame.bad, actual };
}

export function corruptFrame(frame: WalFrame): WalFrame {
  return { ...frame, crc: (frame.crc + 1) >>> 0, bad: true };
}
