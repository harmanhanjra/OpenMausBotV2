// Fan-in event bus — port of upstream's ProviderService fan-in +
// EventNdjsonLogger tee, minus Effect. Every adapter's event stream merges
// into one bus; each event is stamped with its providerInstanceId, teed to
// a per-thread canonical NDJSON log (the debugging trick both upstream and
// agentcal lean on), and delivered to subscribers (the SSE endpoint and
// the server-side message folder).
import { appendFileSync } from "node:fs";
import { join } from "node:path";

import { EVENTS_DIR } from "../config.ts";
import type { ProviderInstance, RuntimeEvent, RuntimeEventListener } from "../contracts.ts";

export class EventBus {
  private listeners = new Set<RuntimeEventListener>();
  private unsubscribes: Array<() => void> = [];
  /** the event log fails per event; one line per process is enough */
  private loggedWriteFailure = false;

  attach(instances: ProviderInstance[]) {
    for (const instance of instances) {
      const unsub = instance.adapter.onEvent((event) => {
        // hard invariant borrowed from correlateRuntimeEventWithInstance:
        // an adapter may only emit events for its own driver kind
        if (event.provider !== instance.driverKind) {
          console.error(`bus: dropped cross-driver event from ${instance.instanceId}`);
          return;
        }
        this.publish({ ...event, providerInstanceId: instance.instanceId });
      });
      this.unsubscribes.push(unsub);
    }
  }

  publish(event: RuntimeEvent) {
    try {
      appendFileSync(join(EVENTS_DIR, `${event.threadId}.ndjson`), JSON.stringify(event) + "\n");
    } catch (e) {
      // logging must never take down the stream, but a log that stopped
      // recording silently makes every later bug report unanswerable
      if (!this.loggedWriteFailure) {
        this.loggedWriteFailure = true;
        console.error(`bus: cannot write the event log in ${EVENTS_DIR} (further failures are not repeated) —`, e);
      }
    }
    for (const listener of [...this.listeners]) {
      try {
        listener(event);
      } catch (e) {
        console.error("bus: listener threw", e);
      }
    }
  }

  subscribe(listener: RuntimeEventListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  detachAll() {
    for (const unsub of this.unsubscribes.splice(0)) unsub();
  }
}
