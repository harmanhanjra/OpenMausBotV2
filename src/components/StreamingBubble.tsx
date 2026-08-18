// The assistant bubble mid-stream, plus the boundary that keeps a bad
// markdown node from taking the app down with it. Both the 1:1 chat and the
// group transcript render the same bubble, and each used to keep its own copy.
import { Component, useDeferredValue, type ReactNode } from "react";

import { ChatMarkdown } from "./ChatMarkdown";

/** One bad markdown node must not white-screen the app — the transcript
 * degrades to a plain-text bubble instead. */
export class MessageBoundary extends Component<
  { children: ReactNode; fallbackText: string },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (this.state.failed) {
      return (
        <div className="max-w-[70%] rounded-2xl bg-card px-4 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-ink">
          {this.props.fallbackText}
        </div>
      );
    }
    return this.props.children;
  }
}

export function StreamingBubble({ text }: { text: string }) {
  // markdown re-parses on a deferred value: when tokens arrive faster than
  // the parser keeps up, React lags the parse instead of janking the frame
  const deferred = useDeferredValue(text);
  return (
    <div className="flex w-full justify-start">
      <div className="max-w-[70%] rounded-2xl bg-card px-4 py-2.5 text-[15px] leading-relaxed text-ink">
        <MessageBoundary fallbackText={deferred}>
          <ChatMarkdown text={deferred} streaming />
        </MessageBoundary>
        <span className="animate-caret ml-0.5 inline-block h-[14px] w-[2px] bg-ink align-middle" />
      </div>
    </div>
  );
}
