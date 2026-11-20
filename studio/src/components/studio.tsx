import { useEffect, useRef, useState, type FormEvent } from "react";
import { BenchPanel } from "@/components/bench-panel";
import { PROMPTS } from "@/lib/microrag/corpus";
import { Engine, formatUs, type Snap } from "@/lib/microrag/engine";
import { runScriptedDemo, type DemoLine } from "@/lib/microrag/scripted";

const STAGES = [
  { title: "Embed", idle: "The question becomes a vector in the same space as the notes." },
  { title: "Cache", idle: "If it is close to a remembered question, the stored answer comes back." },
  { title: "Index", idle: "Otherwise a KD-tree or a flat scan returns the nearest notes." },
  { title: "Answer", idle: "The top note is the completion. The question is kept for next time." },
];

export function Studio() {
  const engineRef = useRef<Engine | null>(null);
  if (engineRef.current === null) engineRef.current = Engine.boot();
  const engine = engineRef.current;

  const [snap, setSnap] = useState<Snap>(() => engine.snapshot());
  const [draft, setDraft] = useState("");
  const [note, setNote] = useState("");
  const [shown, setShown] = useState(0);
  const [live, setLive] = useState(false);
  const [demoLines, setDemoLines] = useState<DemoLine[]>([]);
  const [demoCursor, setDemoCursor] = useState(0);

  useEffect(() => {
    setLive(true);
    setSnap(engine.snapshot());
  }, [engine]);

  useEffect(() => {
    if (!snap.trace) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setShown(4);
      return;
    }
    const timers = [1, 2, 3, 4].map((step) => window.setTimeout(() => setShown(step), 150 * step));
    return () => timers.forEach((id) => window.clearTimeout(id));
  }, [snap.trace?.id]);

  useEffect(() => {
    if (demoCursor <= 0 || demoCursor >= demoLines.length) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const delay = reduce ? 0 : 480;
    const id = window.setTimeout(() => setDemoCursor((c) => c + 1), delay);
    return () => window.clearTimeout(id);
  }, [demoCursor, demoLines.length]);

  function commit() {
    setSnap(engine.snapshot());
  }

  function ask(text: string, follow: string | null) {
    const clean = text.trim();
    if (!clean) return;
    setDraft(clean);
    setShown(0);
    engine.query(clean, follow);
    commit();
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    const known = PROMPTS.find((item) => item.q === draft.trim());
    ask(draft, known?.follow ?? null);
  }

  function playDemo() {
    const lines = runScriptedDemo(0.92);
    setDemoLines(lines);
    setDemoCursor(1);
  }

  const trace = snap.trace;
  const rate = snap.lookups === 0 ? 0 : snap.cacheHits / snap.lookups;

  return (
    <div className="min-h-screen bg-bg text-fg">
      <div className="h-1 bg-copper" />
      <header className="border-b border-muted/25">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <div className="min-w-0">
            <p className="font-mono text-lg font-medium tracking-tight">micro-rag</p>
            <p className="text-sm text-muted">In-memory vector engine</p>
          </div>
          <a
            href="https://github.com/tahaenasraoui-debug/micro-rag"
            className="shrink-0 rounded-md border border-muted/30 px-3 py-2 font-mono text-xs tracking-widest text-fg uppercase"
          >
            Rust source
          </a>
        </div>
      </header>

      <main className="mx-auto flex max-w-6xl flex-col gap-14 px-4 py-8 sm:px-6 sm:py-12">
        <section className="flex flex-col gap-6">
          <div className="max-w-3xl">
            <p className="font-mono text-xs tracking-widest text-copper uppercase">Browser port of the library</p>
            <h1 className="mt-3 text-3xl font-medium tracking-tight sm:text-5xl">
              Same question, different words. Don’t pay for the answer twice.
            </h1>
            <p className="mt-4 max-w-2xl text-base text-muted sm:text-lg">
              Ask the notes. A cold question searches a KD-tree. A paraphrase that clears the cosine
              bar is served from an LRU cache and never touches the index. Every insert is also a
              checksummed log line.
            </p>
          </div>

          <form onSubmit={onSubmit} className="rounded-md border border-muted/25 bg-surface p-4 sm:p-6">
            <label htmlFor="question" className="font-mono text-xs tracking-widest text-muted uppercase">
              Ask the notes
            </label>
            <div className="mt-2 flex flex-col gap-2 sm:flex-row">
              <input
                id="question"
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                placeholder="How do seccomp filters isolate processes?"
                autoComplete="off"
                className="min-w-0 flex-1 rounded-md border border-muted/30 bg-bg px-3 py-3 text-base text-fg"
              />
              <button
                type="submit"
                disabled={!draft.trim()}
                className="rounded-md bg-copper px-5 py-3 text-base font-medium text-ink disabled:opacity-50"
              >
                Search
              </button>
            </div>

            <div className="mt-3 flex flex-wrap gap-2">
              {PROMPTS.map((item) => (
                <button
                  key={item.q}
                  type="button"
                  onClick={() => ask(item.q, item.follow)}
                  className="max-w-full rounded-md border border-muted/30 px-3 py-2 text-left text-sm text-fg"
                >
                  {item.q}
                </button>
              ))}
            </div>

            <div className="mt-5 grid gap-5 sm:grid-cols-3">
              <label className="flex flex-col gap-2 text-sm">
                <span className="font-mono text-xs tracking-widest text-muted uppercase">
                  Cache threshold {snap.threshold.toFixed(2)}
                </span>
                <input
                  type="range"
                  min={0.5}
                  max={0.99}
                  step={0.01}
                  value={snap.threshold}
                  onChange={(event) => {
                    engine.setThreshold(Number(event.target.value));
                    commit();
                  }}
                />
              </label>
              <fieldset>
                <legend className="font-mono text-xs tracking-widest text-muted uppercase">Top notes</legend>
                <div className="mt-2 flex gap-2">
                  {[1, 2, 3].map((k) => (
                    <button
                      key={k}
                      type="button"
                      aria-pressed={snap.topK === k}
                      onClick={() => {
                        engine.setTopK(k);
                        commit();
                      }}
                      className={
                        snap.topK === k
                          ? "min-w-11 rounded-md bg-copper px-3 py-2 text-sm font-medium text-ink"
                          : "min-w-11 rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
                      }
                    >
                      {k}
                    </button>
                  ))}
                </div>
              </fieldset>
              <fieldset>
                <legend className="font-mono text-xs tracking-widest text-muted uppercase">Index</legend>
                <div className="mt-2 flex gap-2" role="radiogroup" aria-label="Index type">
                  {(
                    [
                      ["kdtree", "KD-tree"],
                      ["flat", "Flat"],
                    ] as const
                  ).map(([id, label]) => (
                    <button
                      key={id}
                      type="button"
                      role="radio"
                      aria-checked={snap.indexType === id}
                      onClick={() => {
                        engine.setIndexType(id);
                        commit();
                      }}
                      className={
                        snap.indexType === id
                          ? "rounded-md bg-copper px-3 py-2 text-sm font-medium text-ink"
                          : "rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
                      }
                    >
                      {label}
                    </button>
                  ))}
                </div>
              </fieldset>
            </div>

            <ol className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              {STAGES.map((stage, index) => {
                const open = !trace || shown > index;
                const body = !trace || !open ? stage.idle : stageBody(trace, index);
                const hot =
                  !!trace &&
                  open &&
                  ((index === 1 && trace.outcome === "hit") ||
                    (index === 2 && trace.outcome === "miss") ||
                    index === 3);
                return (
                  <li
                    key={stage.title}
                    className={
                      hot
                        ? "rounded-md border border-copper bg-copper/10 px-3 py-3"
                        : "rounded-md border border-muted/20 px-3 py-3"
                    }
                  >
                    <p className="font-mono text-xs tracking-widest text-muted uppercase">
                      0{index + 1} {stage.title}
                    </p>
                    <p className="mt-2 text-sm text-fg">{body}</p>
                  </li>
                );
              })}
            </ol>

            <div aria-live="polite" className="mt-6 border-t border-muted/20 pt-5">
              {!trace || shown < 4 ? (
                <p className="text-muted">
                  {trace ? "Checking the cache, then the index." : "Pick a question, then ask it a second way."}
                </p>
              ) : (
                <div className="flex flex-col gap-4">
                  <p className="font-mono text-xs tracking-widest text-copper uppercase">
                    {trace.outcome === "hit" ? "Semantic cache hit" : "Cache miss, index read"}
                  </p>
                  <p className="text-lg text-fg">{trace.answer}</p>
                  {trace.outcome === "hit" && trace.matchedQuery && (
                    <p className="text-sm text-muted">
                      Matched “{trace.matchedQuery}” at cosine {trace.similarity.toFixed(3)}, above{" "}
                      {trace.threshold.toFixed(2)}. The index was not read.
                    </p>
                  )}
                  {trace.outcome === "miss" && (
                    <ul className="flex flex-col gap-3">
                      {trace.hits.map((hit) => (
                        <li key={hit.id} className="grid gap-1">
                          <div className="flex items-baseline justify-between gap-3">
                            <span className="font-mono text-xs tracking-widest text-muted uppercase">
                              {hit.id} · {hit.domain}
                            </span>
                            <span className="font-mono text-sm tabular-nums text-fg">{hit.score.toFixed(3)}</span>
                          </div>
                          <div className="h-1.5 rounded-sm bg-fg/10">
                            <div
                              className="h-full rounded-sm bg-copper"
                              style={{ width: `${Math.max(0, Math.min(1, hit.score)) * 100}%` }}
                            />
                          </div>
                          <p className="text-sm text-muted">{hit.text}</p>
                        </li>
                      ))}
                    </ul>
                  )}
                  {trace.follow && (
                    <button
                      type="button"
                      onClick={() => ask(trace.follow as string, null)}
                      className="rounded-md border border-copper px-4 py-3 text-left text-sm text-fg"
                    >
                      <span className="font-mono text-xs tracking-widest text-copper uppercase">Ask the paraphrase</span>
                      <span className="mt-1 block">{trace.follow}</span>
                    </button>
                  )}
                </div>
              )}
            </div>
          </form>
        </section>

        <section className="grid gap-6 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <h2 className="text-xl font-medium tracking-tight">Where the question landed</h2>
            <p className="mt-2 text-sm text-muted">
              Notes sit on the topic their vector activates. Filled dots were retrieved. The ring is
              the question. A cache hit leaves the notes unmarked, because the tree was skipped.
            </p>
            <div className="relative mt-4 h-80 overflow-hidden rounded-md border border-muted/25 bg-surface">
              <div className="pointer-events-none absolute inset-8 rounded-full border border-muted/20" />
              {live ? (
                <>
                  {snap.points.map((point) => (
                    <span
                      key={point.id}
                      title={`${point.id}: ${point.domain}`}
                      className={
                        point.score !== null
                          ? "absolute size-3 rounded-full bg-copper"
                          : "absolute size-2.5 rounded-full bg-fg/70"
                      }
                      style={{ left: `${point.x}%`, top: `${point.y}%`, transform: "translate(-50%, -50%)" }}
                    />
                  ))}
                  {snap.queryPoint && (
                    <span
                      className="absolute z-10 size-4 rounded-full border-2 border-copper bg-bg"
                      style={{
                        left: `${snap.queryPoint.x}%`,
                        top: `${snap.queryPoint.y}%`,
                        transform: "translate(-50%, -50%)",
                      }}
                    />
                  )}
                  <DomainLabels points={snap.points} />
                </>
              ) : (
                <p className="p-4 text-sm text-muted">Projecting the notes.</p>
              )}
            </div>
          </div>

          <div className="flex flex-col gap-3 lg:col-span-2">
            <h2 className="text-xl font-medium tracking-tight">LRU arena</h2>
            <p className="text-sm text-muted tabular-nums">
              {snap.cacheHits} hits / {snap.lookups} lookups · {(rate * 100).toFixed(0)}% · capacity 6, most recent first
            </p>
            {snap.cache.length === 0 ? (
              <p className="rounded-md border border-muted/25 bg-surface p-4 text-sm text-muted">
                Empty. The first search is always a miss. The second wording, if it is close enough, is a hit.
              </p>
            ) : (
              <ol className="flex flex-col gap-2">
                {snap.cache.map((entry, index) => (
                  <li key={`${entry.query}-${index}`} className="rounded-md border border-muted/25 bg-surface px-3 py-3">
                    <p className="font-mono text-xs tracking-widest text-copper uppercase">
                      {entry.mru ? "Most recent" : `Older ${index}`} · {entry.hits} hits
                    </p>
                    <p className="mt-1 text-sm text-fg">{entry.query}</p>
                  </li>
                ))}
              </ol>
            )}
          </div>
        </section>

        <section className="flex flex-col gap-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-xl font-medium tracking-tight">Notes in the index</h2>
              <p className="mt-2 max-w-2xl text-sm text-muted">
                {snap.docs.length} documents. Add one and it is embedded, appended to the log, and folded into the tree.
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                engine.reset();
                setDraft("");
                setNote("");
                setShown(0);
                commit();
              }}
              className="rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
            >
              Restore original notes
            </button>
          </div>
          <ul className="grid gap-3 sm:grid-cols-2">
            {snap.docs.map((doc) => (
              <li key={doc.id} className="rounded-md border border-muted/25 bg-surface p-4">
                <p className="font-mono text-xs tracking-widest text-muted uppercase">
                  {doc.id} · {doc.domain}
                </p>
                <p className="mt-2 text-sm text-fg">{doc.text}</p>
              </li>
            ))}
          </ul>
          <form
            className="flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              engine.addNote(note.slice(0, 600));
              setNote("");
              commit();
            }}
          >
            <label className="sr-only" htmlFor="note">
              Add a note
            </label>
            <input
              id="note"
              value={note}
              maxLength={600}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Add a note, then ask about it"
              className="min-w-0 flex-1 rounded-md border border-muted/30 bg-surface px-3 py-3 text-base text-fg"
            />
            <button
              type="submit"
              disabled={!note.trim()}
              className="rounded-md bg-copper px-4 py-3 text-sm font-medium text-ink disabled:opacity-50"
            >
              Append note
            </button>
          </form>
        </section>

        <section className="flex flex-col gap-4">
          <div>
            <h2 className="text-xl font-medium tracking-tight">The README demo</h2>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              The CLI sample, unchanged: three hand-built 4-D vectors, a cold query, then a paraphrase above 0.92.
              This path does not use the hashing embedder.
            </p>
          </div>
          <button
            type="button"
            onClick={playDemo}
            className="self-start rounded-md bg-copper px-4 py-3 text-sm font-medium text-ink"
          >
            Run the README demo
          </button>
          <pre className="min-h-40 max-w-full overflow-x-auto rounded-md border border-muted/25 bg-bg p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap text-fg sm:text-sm">
            {demoLines.length === 0
              ? "Waiting. This prints the same three steps as `micro-rag demo`."
              : demoLines.slice(0, demoCursor).map((line) => line.text).join("\n\n")}
          </pre>
        </section>

        <BenchPanel />

        <section className="flex flex-col gap-4">
          <div>
            <h2 className="text-xl font-medium tracking-tight">Write-ahead log</h2>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              Each upsert is a JSON frame with an IEEE CRC32, the same checksum as the Rust WAL. Corrupt the
              last frame, then replay. Memory stays intact until replay stops on the bad checksum.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => {
                engine.corruptLast();
                commit();
              }}
              className="rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
            >
              Corrupt last frame
            </button>
            <button
              type="button"
              onClick={() => {
                engine.replay();
                commit();
              }}
              className="rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
            >
              Replay log
            </button>
            <button
              type="button"
              onClick={() => {
                engine.compact();
                commit();
              }}
              className="rounded-md border border-muted/30 px-3 py-2 text-sm text-fg"
            >
              Compact
            </button>
          </div>
          {snap.replayError && (
            <p className="rounded-md border border-copper bg-copper/10 px-3 py-3 text-sm text-fg">{snap.replayError}</p>
          )}
          <ol className="max-h-80 overflow-auto rounded-md border border-muted/25 bg-surface">
            {snap.wal.map((frame) => (
              <li
                key={`${frame.n}-${frame.id}-${frame.crc}`}
                className="flex flex-col gap-1 border-b border-muted/15 px-3 py-3 last:border-b-0 sm:flex-row sm:items-baseline sm:justify-between"
              >
                <p className="min-w-0 text-sm text-fg">
                  <span className="font-mono text-xs text-muted">#{frame.n}</span> {frame.id}{" "}
                  <span className="text-muted">· {frame.domain}</span>
                  {frame.bad ? <span className="text-copper"> · checksum broken</span> : null}
                </p>
                <p className="font-mono text-xs tabular-nums text-muted">crc {frame.crc.toString(16).padStart(8, "0")}</p>
              </li>
            ))}
          </ol>
        </section>

        <footer className="border-t border-muted/25 pt-6 text-sm text-muted">
          Algorithms follow the MIT-licensed Rust crate: cosine kernels, KD-tree pruning after unit
          normalization, an arena LRU cache, and CRC32 frames. The hashing embedder is only here so a typed
          question has a vector. The README demo uses the original 4-D samples.
        </footer>
      </main>
    </div>
  );
}

function stageBody(trace: NonNullable<Snap["trace"]>, index: number): string {
  if (index === 0) return `Vector ready in ${formatUs(trace.us.embed)}.`;
  if (index === 1) {
    if (trace.outcome === "hit") {
      return `Hit ${trace.similarity.toFixed(3)} ≥ ${trace.threshold.toFixed(2)} in ${formatUs(trace.us.cache)}.`;
    }
    if (trace.similarity < 0) return `Miss. Nothing stored yet. ${formatUs(trace.us.cache)}.`;
    return `Miss. Closest was ${trace.similarity.toFixed(3)}, under ${trace.threshold.toFixed(2)}. ${formatUs(trace.us.cache)}.`;
  }
  if (index === 2) {
    if (trace.outcome === "hit") return "Skipped.";
    const kind = trace.indexType === "kdtree" ? "KD-tree" : "Flat scan";
    if (trace.indexType === "kdtree" && trace.scanned >= trace.total) {
      return `${kind} read all ${trace.total} notes in ${formatUs(trace.us.search)}. Too few to prune.`;
    }
    return `${kind} read ${trace.scanned} of ${trace.total} in ${formatUs(trace.us.search)}.`;
  }
  return trace.outcome === "hit" ? "Returned the stored completion." : "Stored the new completion in the arena.";
}

function DomainLabels({ points }: { points: Snap["points"] }) {
  const groups = new Map<string, { x: number; y: number; n: number }>();
  for (const point of points) {
    const group = groups.get(point.domain) ?? { x: 0, y: 0, n: 0 };
    group.x += point.x;
    group.y += point.y;
    group.n += 1;
    groups.set(point.domain, group);
  }
  return (
    <>
      {[...groups.entries()].map(([domain, group]) => {
        const cx = group.x / group.n;
        const cy = group.y / group.n;
        const dx = cx - 50;
        const dy = cy - 50;
        const len = Math.hypot(dx, dy) || 1;
        const lx = Math.min(88, Math.max(12, cx + (dx / len) * 8));
        const ly = Math.min(90, Math.max(8, cy + (dy / len) * 8));
        return (
          <span
            key={domain}
            className="pointer-events-none absolute font-mono text-xs tracking-widest text-fg uppercase"
            style={{ left: `${lx}%`, top: `${ly}%`, transform: "translate(-50%, -50%)" }}
          >
            {domain}
          </span>
        );
      })}
    </>
  );
}
