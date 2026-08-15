"use client";

type Item = {
  key: string;
  label: string;
  active: boolean;
  onPress: () => void;
};

export function TopPillBar({ items }: { items: Item[] }) {
  return (
    <div className="top-pill-bar">
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          onClick={item.onPress}
          className={item.active ? "top-pill top-pill-active" : "top-pill"}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}
