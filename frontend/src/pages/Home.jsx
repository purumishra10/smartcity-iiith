import { Link } from "react-router-dom";

const STEPS = [
  {
    n: "01",
    title: "Submit a still",
    text: "Upload a street photograph, CCTV grab, or incident image. We accept JPEG, PNG, WebP, and BMP files up to 10 MB.",
  },
  {
    n: "02",
    title: "Measure the frame",
    text: "Classical computer vision scores sharpness, exposure, contrast, noise, and local anomalies — the same checks an operator would make by eye, but consistently.",
  },
  {
    n: "03",
    title: "A local model decides",
    text: "A hybrid network (classical vitals plus a small CNN) runs on this machine. It estimates overall quality and six issue types. Nothing is sent to an external AI service.",
  },
  {
    n: "04",
    title: "Review the report",
    text: "You receive a score, a plain-language diagnosis, issue confidence, and heatmaps that show where problems sit on the frame. Create an account if you want that exam kept in Past exams.",
  },
];

const ISSUES = [
  { title: "Blur", text: "Motion or focus loss that hides number plates, faces, or signage." },
  { title: "Underexposure", text: "Frames crushed into shadow, typical of night or backlit scenes." },
  { title: "Overexposure", text: "Blown highlights that erase headlights, sky, or reflective surfaces." },
  { title: "Noise", text: "Grain and sensor noise that reduce usable detail in low light." },
  { title: "Corruption", text: "Severe JPEG smash or decode damage that makes a file unusable." },
  { title: "Visual defect", text: "Local stains, scratches, or patches unlike the rest of the image." },
];

export default function Home() {
  return (
    <div className="home">
      <section className="hero">
        <div className="home-inner">
          <p className="hero-kicker">Civic still inspection</p>
          <h1 className="hero-title">Know if a public-space image is fit to use — before you act on it.</h1>
          <p className="hero-lead">
            Dr. Image reviews street and camera stills for blur, exposure, noise, corruption, and
            visual defects. It returns a quality score, a diagnosis, and maps of where the problems are.
            Analysis runs entirely on your infrastructure.
          </p>
          <div className="hero-actions">
            <Link className="btn" to="/intake">
              Start an exam
            </Link>
            <a className="btn-ghost" href="#how-it-works">
              See how it works
            </a>
          </div>
        </div>
      </section>

      <section className="stat-strip" aria-label="Product facts">
        <div className="home-inner stat-grid">
          <div>
            <strong>6</strong>
            <span>Quality issues screened on every upload</span>
          </div>
          <div>
            <strong>On-device</strong>
            <span>No third-party vision APIs or API keys</span>
          </div>
          <div>
            <strong>Mapped</strong>
            <span>Region heatmaps, not just a single number</span>
          </div>
          <div>
            <strong>Optional save</strong>
            <span>Free to run; account required to keep Past exams</span>
          </div>
        </div>
      </section>

      <section className="home-block" id="how-it-works">
        <div className="home-inner">
          <p className="section-kicker">Process</p>
          <h2>From upload to a decision you can explain</h2>
          <p className="section-lead">
            Dr. Image is built for operators who need a defensible quality check — not a black-box score.
          </p>
          <ol className="step-grid">
            {STEPS.map((s) => (
              <li key={s.n}>
                <span className="step-n">{s.n}</span>
                <h3>{s.title}</h3>
                <p>{s.text}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="home-block home-block--alt" id="detections">
        <div className="home-inner">
          <p className="section-kicker">Coverage</p>
          <h2>What every exam looks for</h2>
          <p className="section-lead">
            Required quality problems are first-class. Each finding carries severity and a confidence score.
          </p>
          <ul className="issue-grid">
            {ISSUES.map((item) => (
              <li key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="home-block" id="method">
        <div className="home-inner split">
          <div>
            <p className="section-kicker">Method</p>
            <h2>Measurements first. A learned model second. A written verdict last.</h2>
            <p>
              We extract interpretable vitals from the image — sharpness, brightness, contrast, noise, and
              local residuals. A hybrid model (classical features plus a small CNN) turns those signals into a
              0–100 score and issue probabilities. Simple rules then assign ACCEPTABLE, DEGRADED, or
              DEFECTIVE, and a written diagnosis in plain language.
            </p>
            <p>
              Heatmaps reuse the same tile-level measurements as the global score, so a highlighted region
              is not an unrelated overlay. Click a hot spot on the exam page to inspect that cell.
            </p>
          </div>
          <aside className="method-card">
            <h3>What you get back</h3>
            <ul>
              <li>Quality score and label</li>
              <li>Primary diagnosis</li>
              <li>Issues with severity and confidence</li>
              <li>Blur, exposure, noise, and defect maps</li>
              <li>A downloadable PDF with original + four heatmaps</li>
              <li>Past exams when you create a free account</li>
            </ul>
          </aside>
        </div>
      </section>

      <section className="home-cta">
        <div className="home-inner">
          <h2>Ready to inspect a still?</h2>
          <p>Open intake and run a local quality exam. Sign up only if you want Past exams saved.</p>
          <Link className="btn" to="/intake">
            Go to intake
          </Link>
        </div>
      </section>
    </div>
  );
}
