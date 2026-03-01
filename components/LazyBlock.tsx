import type { ComponentChildren, JSX } from 'preact';

const lazyBlockStyle: JSX.CSSProperties = {
  contentVisibility: 'auto' as any,
  containIntrinsicSize: 'auto 200px' as any,
};

// Skip rendering off-screen EPUB/text content via CSS content-visibility:auto.
// Keeps the DOM intact for Cmd+F and getElementById-based scroll-to-resume.
const LazyBlock = ({ children }: { children: ComponentChildren }) => (
  <div style={lazyBlockStyle}>{children}</div>
);

export default LazyBlock;
