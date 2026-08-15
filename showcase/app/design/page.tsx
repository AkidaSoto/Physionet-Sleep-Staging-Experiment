import { Button } from "../../components/ui/Button";
import { InsetSurface, Surface } from "../../components/ui/Surface";
import { Text } from "../../components/ui/Text";
import { colors } from "../../design/tokens";

const tokenEntries = Object.entries(colors);

export default function DesignPage() {
  return (
    <main className="page">
      <section className="hero-block">
        <Text as="h1" variant="display">
          Showcase primitives
        </Text>
        <Text style={{ maxWidth: 760 }}>
          This page is the primitive lab: tokens, text roles, surfaces, buttons,
          and reference affordances. It should stay low-drama and easy to reskin.
        </Text>
      </section>

      <Surface elevated style={{ marginBottom: 16 }}>
        <Text as="h2" variant="title">
          Typography
        </Text>
        <div className="grid" style={{ marginTop: 16 }}>
          <Text as="div" variant="display">
            Display
          </Text>
          <Text as="div" variant="title">
            Title
          </Text>
          <Text as="div" variant="sectionTitle">
            Section title
          </Text>
          <Text as="div" variant="body">
            Body text explains ideas without feeling noisy.
          </Text>
          <Text as="div" variant="bodyStrong">
            Strong body text is for emphasis without shouting.
          </Text>
          <Text as="div" variant="meta" tone="secondary">
            Meta text is secondary and supportive.
          </Text>
        </div>
      </Surface>

      <div className="grid two" style={{ marginBottom: 16 }}>
        <Surface>
          <Text as="h2" variant="title">
            Tokens
          </Text>
          <div className="token-list">
            {tokenEntries.map(([key, value]) => (
              <div className="token-row" key={key}>
                <div className="token-swatch" style={{ background: value }} />
                <div>
                  <Text as="div" variant="bodyStrong">
                    {key}
                  </Text>
                  <Text as="div" variant="meta" tone="secondary">
                    {value}
                  </Text>
                </div>
              </div>
            ))}
          </div>
        </Surface>

        <Surface>
          <Text as="h2" variant="title">
            Primitive modules
          </Text>
          <div className="grid" style={{ marginTop: 16 }}>
            <InsetSurface>
              <Text as="h3" variant="sectionTitle">
                Surface + inset surface
              </Text>
              <Text tone="secondary">
                Main cards hold sections. Inset surfaces hold tighter grouped
                content inside them.
              </Text>
            </InsetSurface>

            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Button>Primary action</Button>
              <Button variant="ghost">Ghost action</Button>
            </div>
          </div>
        </Surface>
      </div>

      <Surface>
        <Text as="h2" variant="title">
          Reference affordance
        </Text>
        <Text tone="secondary" style={{ marginTop: 8, marginBottom: 16 }}>
          Research context should be attached as a small, consistent hint rather
          than baked into every page header.
        </Text>
        <div className="reference-item" style={{ maxWidth: 480 }}>
          <span className="q-badge">?</span>
          <div className="reference-copy">
            <strong>UCDDB dataset</strong>
            <span className="muted">
              Primary PhysioNet source for signals, records, and annotations.
            </span>
          </div>
        </div>
      </Surface>
    </main>
  );
}
