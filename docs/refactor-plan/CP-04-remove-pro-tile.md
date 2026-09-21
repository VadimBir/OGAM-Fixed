# CP-04 — Remove the "Off Grid Pro" tile from Home

Read first: `docs/refactor-map/04-cards-personas.md` (Pro affordance inventory).

## USER DECISION (2026-09-20)
The annoyance = the "get into Pro" TOP TILE in Settings = `<ProUpsellBanner>`
(`SettingsScreen.tsx` L146-149). Remove it. That location is later reused for the
tool/skill CARDS (see CP-06/CP-07). Keep the functional `proNavButton` (license status)
for now. Home crown (`HomeScreen` L190-198) is a secondary candidate — confirm before
removing. Settings ALSO needs a SEPARATED, EDITABLE system-instruction element → handled
by the new "System" tab (CP-06/07), not here.

## Settings tab structure the user wants (from answers)
Existing Model/Generation settings have Image-gen + Text-gen tabs. Add a new **System**
tab with sub-tabs: System Prompt (viewable+editable), Skills/Tool Cards, Characters,
Persona (user persona, sibling of Characters). See CP-06 & CP-07.

## Candidates on HomeScreen (`src/screens/HomeScreen/index.tsx`)
- Crown button, L190-198: `navigation.navigate('ProDetail')`, a11y "Open Off Grid AI Pro".
- `SyncHomeCard` slot (L256+) — Pro-gated sync card.
- `DesktopPromoCard` (L328).

## Plan (pending user confirm on WHICH element)
- If crown button: remove the `<TouchableOpacity>` block L190-198 and its `crownButton`
  style; keep `HomeNotificationsButton`. Ensure `ProDetail` still reachable elsewhere
  (Settings) or confirm user wants it fully gone.
- Keep pro/ submodule code intact; this is a Home UI removal only.

## Verification
- Screenshot Home before/after on device. `npx tsc --noEmit`, jest for HomeScreen RNTL
  test if present. No dangling unused imports (crownButton style, IconMC).

## Open question for user
The crown button, the Sync/Pro card, or the Desktop promo card? (Screenshot annotation
would resolve instantly.)
