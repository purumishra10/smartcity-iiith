export default function BrandMark({ size = 36 }) {
  return (
    <svg
      className="brand-mark"
      width={size}
      height={size}
      viewBox="0 0 64 64"
      aria-hidden="true"
    >
      <rect width="64" height="64" rx="10" fill="#1c2430" />
      <path d="M8 46 L18 34 L28 40 L42 22 L56 36" fill="none" stroke="#7eb3d9" strokeWidth="2.4" />
      <path d="M8 52 H56" stroke="#3d4a5c" strokeWidth="1.5" />
      <rect x="10" y="40" width="7" height="12" fill="#5a6a7c" />
      <rect x="20" y="34" width="8" height="18" fill="#6a7c90" />
      <rect x="31" y="28" width="7" height="24" fill="#5a6a7c" />
      <circle cx="44" cy="22" r="11" fill="none" stroke="#e8e4dc" strokeWidth="2.4" />
      <circle cx="44" cy="22" r="4.5" fill="none" stroke="#c9a227" strokeWidth="2" />
      <path d="M52 30 L58 36" stroke="#e8e4dc" strokeWidth="2.4" strokeLinecap="round" />
      <rect x="29" y="8" width="6" height="16" rx="1" fill="#c44b3c" />
      <rect x="24" y="13" width="16" height="6" rx="1" fill="#c44b3c" />
    </svg>
  );
}
