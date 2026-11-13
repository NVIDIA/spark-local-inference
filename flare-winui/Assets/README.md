# Windows application icon

The cream block `h` and ascending solar plume refer to the
[upstream Hyperflare banner](https://github.com/NVIDIA/hyperflare-inference/blob/main/docs/images/hyperflare.png)
linked from the [repository README](https://github.com/NVIDIA/hyperflare-inference).
The icon is a separate vector interpretation; it contains no banner pixels.
The plume uses NVIDIA green `#76B900`. The tile, bevel, border, and letter shadow
use neutral charcoal. Outer corners are transparent.

- `flare-app.svg`: scalable artwork, including the large-size highlight and shadow.
- `flare-app-small.svg`: simplified artwork for 16–32 px frames.
- `flare-app.ico`: nine 32-bit PNG-compressed icon frames at 16, 20, 24, 32, 40,
  48, 64, 128, and 256 px.
- `flare-app.png`: 512 px transparent preview.
- `flare-app-preview.png`: native-size checks on dark and light backgrounds.
- `flare-app-small-preview.png`: actual and enlarged 16, 24, and 32 px frames.
- `flare-app-{size}.png`: individual transparent icon frames.
- `branding-reference.png`: the downloaded upstream banner, for design review
  only; exclude it and the contact sheets from application deployment.

Run `python flare-winui/Assets/render_icon.py` from the repository root on Windows
with Pillow installed to regenerate the PNGs and ICO from the SVG polygons.
The renderer validates frame dimensions, bounds, and alpha after writing the ICO.
Pillow is an asset-authoring dependency, not an application dependency.

`flare-winui.exe` embeds `flare-app.ico` as the Win32 `IDI_FLARE_APP` resource for
Explorer. The window loads its small and large icons from that resource for the
window and taskbar; no external icon file is required at runtime.
`flare-winui/flare-winui.rc` also embeds `flare-app-128.png` as the
`IDB_FLARE_TITLE` resource, which supplies the title bar image.
`flare-winui/resource.h` defines both identifiers. The single-file launcher
`Flare.exe` embeds `flare-app.ico` as icon resource 1 through
`flare-winui/launcher/build.rs`.
