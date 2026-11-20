import { useState } from "react";
import { runBench, type BenchResult } from "@/lib/microrag/bench";

const SIZES = [1000, 2500, 5000];
const DIMS = [2, 4, 8, 16];

function fmt(n: number): string {
  return n.toFixed(n >= 100 ? 0 : 1);
}

export function BenchPanel() {
  const [size, setSize] = useState(2500);
  const [dim, setDim] = useState(4);
  const [phase, setPhase] = useState<string | null>(null);
  const [result, setResult] = useState<BenchResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function go() {
    setError(null);
    setPhase("Sampling vectors");
    try {
      const next = await runBench(size, dim, 5, 30, setPhase);
      setResult(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Benchmark failed");
    } finally {
      setPhase(null);
    }
  }

  const maxUs = result ? Math.max(result.flatUs, result.kdUs, 0.001) : 1;

  return (
    <section className="flex flex-col gap-4">
      <div>
        <h2 className="text-xl font-medium tracking-tight">Flat scan versus KD-tree</h2>
        <p className="mt-2 max-w-2xl text-sm text-muted">
          Same Xorshift vectors as the Rust bench, exact top-5 recall. The tree prunes in low
          dimension. Past that, the splitting plane almost always cuts the search radius, and it
          reads nearly every node.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <fieldset>
          <legend className="font-mono text-xs tracking-widest text-muted uppercase">Vectors</legend>
          <div className="mt-2 flex flex-wrap gap-2">
            {SIZES.map((n) => (
              <button
                key={n}
                type="button"
                aria-pressed={size === n}
                onClick={() => setSize(n)}
                className={
                  size === n
                    ? "rounded-md bg-copper px-3 py-2 text-sm font-medium text-ink"
                    : "rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
                }
              >
                {n.toString()}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend className="font-mono text-xs tracking-widest text-muted uppercase">Dimensions</legend>
          <div className="mt-2 flex flex-wrap gap-2">
            {DIMS.map((n) => (
              <button
                key={n}
                type="button"
                aria-pressed={dim === n}
                onClick={() => setDim(n)}
                className={
                  dim === n
                    ? "rounded-md bg-copper px-3 py-2 text-sm font-medium text-ink"
                    : "rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
                }
              >
                {n}D
              </button>
            ))}
          </div>
        </fieldset>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={() => void go()}
          disabled={phase !== null}
          className="rounded-md bg-copper px-4 py-3 text-sm font-medium text-ink disabled:opacity-50"
        >
          {phase ?? "Run benchmark"}
        </button>
        <p className="text-sm text-muted">30 queries, top-5, cosine. Times are this browser.</p>
      </div>

      {error && <p className="text-sm text-fg">{error}</p>}

      {result && (
        <div className="grid gap-4 rounded-md border border-muted/25 bg-surface p-4 sm:grid-cols-2">
          <div>
            <p className="font-mono text-xs tracking-widest text-muted uppercase">Flat scan</p>
            <p className="mt-1 font-mono text-2xl tabular-nums text-fg">{fmt(result.flatUs)} µs</p>
            <div className="mt-3 h-2 rounded-sm bg-fg/10">
              <div className="h-full rounded-sm bg-muted" style={{ width: `${(result.flatUs / maxUs) * 100}%` }} />
            </div>
            <p className="mt-2 text-sm text-muted tabular-nums">
              {result.scannedFlat.toString()} node visits · {fmt(result.flatMs)} ms total
            </p>
          </div>
          <div>
            <p className="font-mono text-xs tracking-widest text-muted uppercase">KD-tree</p>
            <p className="mt-1 font-mono text-2xl tabular-nums text-copper">{fmt(result.kdUs)} µs</p>
            <div className="mt-3 h-2 rounded-sm bg-fg/10">
              <div className="h-full rounded-sm bg-copper" style={{ width: `${(result.kdUs / maxUs) * 100}%` }} />
            </div>
            <p className="mt-2 text-sm text-muted tabular-nums">
              {result.scannedKd.toString()} node visits · build {fmt(result.buildMs)} ms · recall{" "}
              {result.recall.toFixed(1)}%
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
