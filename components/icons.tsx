// components/icons.tsx — extracted from lucide-preact v1.17.0 (ISC).
// All 17 icons used by the app; tree-shake friendly single export.

import type { JSX } from "preact";

type IconName = "target"|"loader-circle"|"check"|"play"|"pause"|"menu"|"palette"|"beaker"|"clock"|"trash-2"|"book-open"|"file-text"|"globe"|"text-align-justify"|"x"|"upload"|"clipboard"|"bookmark"|"chevron-left"|"chevron-right"|"volume";
type IconTuple = [string, Record<string, string | number>];
const ICONS: Record<IconName, IconTuple[]> = {
  target: [
    ["circle",{cx:"12", cy:"12", r:"10", key:"1mglay"}],
    ["circle",{cx:"12", cy:"12", r:"6", key:"1vlfrh"}],
    ["circle",{cx:"12", cy:"12", r:"2", key:"1c9p78"}],
  ],
  "loader-circle": [
    ["path",{d:"M21 12a9 9 0 1 1-6.219-8.56", key:"13zald"}],
  ],
  check: [
    ["path",{d:"M20 6 9 17l-5-5", key:"1gmf2c"}],
  ],
  play: [
    ["path",{d:"M5 5a2 2 0 0 1 3.008-1.728l11.997 6.998a2 2 0 0 1 .003 3.458l-12 7A2 2 0 0 1 5 19z", key:"10ikf1"}],
  ],
  pause: [
    ["rect",{x:"14", y:"3", width:"5", height:"18", rx:"1", key:"kaeet6"}],
    ["rect",{x:"5", y:"3", width:"5", height:"18", rx:"1", key:"1wsw3u"}],
  ],
  menu: [
    ["path",{d:"M4 5h16", key:"1tepv9"}],
    ["path",{d:"M4 12h16", key:"1lakjw"}],
    ["path",{d:"M4 19h16", key:"1djgab"}],
  ],
  palette: [
    ["path",{d:"M12 22a1 1 0 0 1 0-20 10 9 0 0 1 10 9 5 5 0 0 1-5 5h-2.25a1.75 1.75 0 0 0-1.4 2.8l.3.4a1.75 1.75 0 0 1-1.4 2.8z", key:"e79jfc"}],
    ["circle",{cx:"13.5", cy:"6.5", r:".5", fill:"currentColor", key:"1okk4w"}],
    ["circle",{cx:"17.5", cy:"10.5", r:".5", fill:"currentColor", key:"f64h9f"}],
    ["circle",{cx:"6.5", cy:"12.5", r:".5", fill:"currentColor", key:"qy21gx"}],
    ["circle",{cx:"8.5", cy:"7.5", r:".5", fill:"currentColor", key:"fotxhn"}],
  ],
  beaker: [
    ["path",{d:"M4.5 3h15", key:"c7n0jr"}],
    ["path",{d:"M6 3v16a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V3", key:"m1uhx7"}],
    ["path",{d:"M6 14h12", key:"4cwo0f"}],
  ],
  clock: [
    ["circle",{cx:"12", cy:"12", r:"10", key:"1mglay"}],
    ["path",{d:"M12 6v6l4 2", key:"mmk7yg"}],
  ],
  "trash-2": [
    ["path",{d:"M10 11v6", key:"nco0om"}],
    ["path",{d:"M14 11v6", key:"outv1u"}],
    ["path",{d:"M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6", key:"miytrc"}],
    ["path",{d:"M3 6h18", key:"d0wm0j"}],
    ["path",{d:"M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2", key:"e791ji"}],
  ],
  "book-open": [
    ["path",{d:"M12 7v14", key:"1akyts"}],
    ["path",{d:"M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z", key:"ruj8y"}],
  ],
  "file-text": [
    ["path",{d:"M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z", key:"1oefj6"}],
    ["path",{d:"M14 2v5a1 1 0 0 0 1 1h5", key:"wfsgrz"}],
    ["path",{d:"M10 9H8", key:"b1mrlr"}],
    ["path",{d:"M16 13H8", key:"t4e002"}],
    ["path",{d:"M16 17H8", key:"z1uh3a"}],
  ],
  globe: [
    ["circle",{cx:"12", cy:"12", r:"10", key:"1mglay"}],
    ["path",{d:"M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20", key:"13o1zl"}],
    ["path",{d:"M2 12h20", key:"9i4pu4"}],
  ],
  "text-align-justify": [
    ["path",{d:"M3 5h18", key:"1u36vt"}],
    ["path",{d:"M3 12h18", key:"1i2n21"}],
    ["path",{d:"M3 19h18", key:"awlh7x"}],
  ],
  x: [
    ["path",{d:"M18 6 6 18", key:"1bl5f8"}],
    ["path",{d:"m6 6 12 12", key:"d8bk6v"}],
  ],
  upload: [
    ["path",{d:"M12 3v12", key:"1x0j5s"}],
    ["path",{d:"m17 8-5-5-5 5", key:"7q97r8"}],
    ["path",{d:"M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4", key:"ih7n3h"}],
  ],
  clipboard: [
    ["rect",{width:"8", height:"4", x:"8", y:"2", rx:"1", ry:"1", key:"tgr4d6"}],
    ["path",{d:"M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2", key:"116196"}],
  ],
  bookmark: [
    ["path",{d:"M17 3a2 2 0 0 1 2 2v15a1 1 0 0 1-1.496.868l-4.512-2.578a2 2 0 0 0-1.984 0l-4.512 2.578A1 1 0 0 1 5 20V5a2 2 0 0 1 2-2z", key:"oz39mx"}],
  ],
  "chevron-left": [
    ["path",{d:"m15 18-6-6 6-6", key:"chevron-left"}],
  ],
  "chevron-right": [
    ["path",{d:"m9 18 6-6-6-6", key:"chevron-right"}],
  ],
  volume: [
    ["path",{d:"M11 5 6 9H2v6h4l5 4V5z", key:"1ytikj"}],
    ["path",{d:"M19.07 4.93a10 10 0 0 1 0 14.14", key:"1d5fby"}],
    ["path",{d:"M15.54 8.46a5 5 0 0 1 0 7.07", key:"1rc4g9"}],
  ],
};

interface Props {
  name: IconName;
  size?: number;
  color?: string;
  fill?: string;
  strokeWidth?: number;
  style?: JSX.CSSProperties;
  className?: string;
  title?: string;
}
export default function Icon({ name, size = 24, color = "currentColor", fill = "none", strokeWidth = 2, style, className, title }: Props) {
  const children = ICONS[name];
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={fill}
      stroke={color}
      stroke-width={strokeWidth}
      strokeLinecap="round"
      strokeLinejoin="round"
      style={style}
      class={className}
      aria-label={title}
    >
      {children.map(([tag, attrs], i) => {
        const c: Record<string, any> = {};
        for (const k in attrs) c[k] = attrs[k];
        const key = String(c.key ?? i);
        const passthrough = { ...c };
        delete (passthrough as any).key;
        switch (tag) {
          case 'path': return <path key={key} {...passthrough} />;
          case 'circle': return <circle key={key} {...passthrough} />;
          case 'line': return <line key={key} {...passthrough} />;
          case 'rect': return <rect key={key} {...passthrough} />;
          case 'polyline': return <polyline key={key} {...passthrough} />;
          case 'polygon': return <polygon key={key} {...passthrough} />;
          case 'ellipse': return <ellipse key={key} {...passthrough} />;
          default: return null;
        }
      })}
    </svg>
  );
}
