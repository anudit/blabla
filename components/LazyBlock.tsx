import type { ComponentChildren, JSX } from 'preact';

interface LazyBlockProps {
  id: string;
  // These are kept in props for future metadata needs, 
  // but we no longer need to "force" renders since we aren't unmounting.
  startLineId?: number;
  endLineId?: number;
  children: ComponentChildren;
}

const lazyBlockStyle: JSX.CSSProperties = {
  // CRITICAL: content-visibility: auto is the "silver bullet".
  // It stops the browser from painting/laying out off-screen blocks,
  // BUT it keeps the text in the DOM so Ctrl+F (browser search) still works perfectly.
  contentVisibility: 'auto' as any,
  // We provide a hint to the browser about the block size to prevent scrollbar jumping.
  containIntrinsicSize: 'auto 100px' as any,
  margin: '0 0 1em 0',
};

const LazyBlock = ({ children }: LazyBlockProps) => {
  return (
    <div style={lazyBlockStyle}>
      {children}
    </div>
  );
};

export default LazyBlock;
