import Link from "next/link";

const items = [
  { href: "/#top", label: "Top" },
  { href: "/#abstract", label: "Abstract" },
  { href: "/#apnea", label: "Apnea" },
  { href: "/#staging", label: "Staging" },
  { href: "/#about", label: "About" }
];

export function SiteNav() {
  return (
    <nav className="site-nav" aria-label="Primary">
      <div className="site-nav-inner">
        <Link href="/#top" className="site-nav-brand">
          Sleep Explorer
        </Link>
        <div className="site-nav-links">
          {items.map((item) => (
            <Link key={item.href} href={item.href} className="site-nav-link">
              {item.label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
