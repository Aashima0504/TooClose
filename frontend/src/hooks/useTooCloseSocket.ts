/** Throttled WebSocket client. Implemented in Phase 8. */
export function useTooCloseSocket(): { status: 'idle' | 'open' | 'offline' } {
  return { status: 'idle' }
}
