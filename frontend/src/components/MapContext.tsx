import { MapPin, Check, RotateCcw } from "lucide-react";
import Help from "./Help";
import RoofSelectionCard from "./RoofSelectionCard";
import type { MapCapture, MapPick } from "../mapTypes";

export default function MapContext({
  capture,
  busy,
  edited,
  onReset,
}: {
  capture: MapCapture;
  onPick: (pick: MapPick) => Promise<void>;
  busy: boolean;
  edited: boolean;
  onReset: () => void;
}) {
  const p = capture.provenance;
  const measured = capture.objects.filter((o) => o.source === "elevation");
  const windows = capture.objects.filter((o) => o.source === "image" && o.kind === "skylight");
  return (
    <section className="map-context">
      <div className="map-context-title">
        <span>
          <MapPin size={16} />
          Roof from the map
        </span>
        <strong>
          <Check size={13} />
          Scale calibrated
        </strong>
      </div>
      <p>{p.address?.label ?? "Address unavailable"} / {p.latitude.toFixed(6)}, {p.longitude.toFixed(6)}</p>
      <div className="map-facts">
        <span>
          {(100 / capture.pixels_per_metre).toFixed(1)} cm / image pixel
        </span>
        <span>{p.crs} · metric capture</span>
        {p.pitch_deg != null && <span>Source roof pitch: {p.pitch_deg}°</span>}
        {!!(measured.length || windows.length) && (
          <span>
            {measured.length} raised · {windows.length} windows
            <Help title="What was found on your roof">
              <b>Raised structures</b> — chimneys, dormers and vents — come from
              swisstopo's height model, which measures the roof surface every
              50 cm. Anything standing more than 28 cm proud of its own roof
              face is kept clear.
              <br />
              <br />
              <b>Roof windows</b> lie flush in the pitch, so no height model can
              see them. These are found in the photo instead: glass reflects the
              sky, so a window reads blue against a red or brown roof.
              <br />
              <br />
              That last one is a colour rule, not a trained model. It can miss a
              window in deep shadow and can be fooled by a blue-grey roof, so
              compare the overlay with the image and mark anything missed.
            </Help>
          </span>
        )}
      </div>
      <p>
        {edited
          ? "Boundary adjusted by you."
          : capture.roof.length > 2 ? "Official outline imported automatically."
          : "No roof outline was returned. Draw the roof or select a point on its surface."}{" "}
        Drag vertices or redraw the roof; mark any missed PV or obstacles, then
        analyse again.
      </p>
      <p>
        Scale comes from the map. Each reliable face is packed in surface metres;
        unavailable faces use a labelled projected fallback. Roof records and aerial
        imagery may differ in age.
      </p>
      {capture.warnings.map((w) => (
        <p key={w}>{w}</p>
      ))}
      {edited && capture.roof.length > 2 && (
        <button className="text-button" disabled={busy} onClick={onReset}>
          <RotateCcw size={13} />
          Restore official outline & alignment
        </button>
      )}
      <RoofSelectionCard capture={capture} />
    </section>
  );
}
