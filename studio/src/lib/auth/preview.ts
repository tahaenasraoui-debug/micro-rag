/**
 * Shared LIVE-PREVIEW OAuth client (server-only — NEVER import from the client).
 *
 * The sandbox serves each live preview on a dynamic `https://*.grok-sandbox.com`
 * URL, which can't be pre-registered per app. The broker instead exposes ONE
 * shared "preview" client that accepts any
 * `https://*.grok-sandbox.com/api/auth/oauth2/callback/*`
 * (broker: `app-builder-deployer/auth/src/preview-oauth.ts`). When deployed the
 * deployer injects a per-app `GROK_AUTH_*` that overrides these (see `server.ts`).
 *
 * Supply `GROK_PREVIEW_CLIENT_SECRET` (and optionally `GROK_PREVIEW_CLIENT_ID`)
 * via environment variables to match the broker's preview client. When
 * `GROK_PREVIEW_CLIENT_SECRET` is unset, `PREVIEW_CLIENT_SECRET` resolves to
 * `undefined` and `server.ts` disables preview OAuth gracefully.
 */
export const PREVIEW_CLIENT_ID =
  process.env.GROK_PREVIEW_CLIENT_ID?.trim() || "grok_preview";
export const PREVIEW_CLIENT_SECRET: string | undefined =
  process.env.GROK_PREVIEW_CLIENT_SECRET?.trim() || undefined;

/** The shared auth broker issuer (OIDC discovery lives under it). */
export const GROK_ISSUER_DEFAULT = "https://auth.grok.me";

/**
 * Host patterns whose callbacks the preview client accepts. Better Auth derives
 * the live preview's real origin from the request host and validates it against
 * this list (wildcard-matched), so the OAuth `redirect_uri` becomes the concrete
 * `https://<preview-host>/api/auth/oauth2/callback/...` the broker allows.
 */
export const PREVIEW_ALLOWED_HOSTS = ["*.grok-sandbox.com"] as const;
