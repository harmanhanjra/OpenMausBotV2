// Runtime-event plumbing every driver shares: a listener set, the fan-out,
// and the envelope (eventId/provider/threadId/turnId/createdAt) that prefixes
// every emitted event.
import type { RuntimeEvent, RuntimeEventListener } from "../contracts.ts";
import { newEventId } from "../contracts.ts";

export interface EventBase {
  eventId: string;
  provider: string;
  threadId: string;
  turnId: string;
  createdAt: string;
  /** e.g. providerInstanceId, for drivers that tag their events with it. */
  [extra: string]: string;
}

export interface EventHub {
  emit(event: RuntimeEvent): void;
  /** Fresh envelope for one event — spread it into the event literal. */
  base(threadId: string, turnId: string): EventBase;
  /** ProviderAdapter.onEvent: subscribe, returns the unsubscribe. */
  onEvent(listener: RuntimeEventListener): () => void;
  /** Drop every listener (driver dispose). */
  clear(): void;
}

/** `extra` fields land in every envelope (prime-agent tags its
 * providerInstanceId that way). */
export function createEventHub(provider: string, extra?: Record<string, string>): EventHub {
  const listeners = new Set<RuntimeEventListener>();
  return {
    emit(event) {
      for (const l of [...listeners]) l(event);
    },
    base(threadId, turnId) {
      return {
        eventId: newEventId(),
        provider,
        ...extra,
        threadId,
        turnId,
        createdAt: new Date().toISOString(),
      };
    },
    onEvent(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    clear() {
      listeners.clear();
    },
  };
}
