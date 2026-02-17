# HALO Hackathon Data Dictionary

Built from `instructions.md`. Joins use `game_id`, `player_id`, and `sl_event_id`.

---

## games

One record per game.

| Field | Description |
|-------|-------------|
| `game_id` | Unique game identifier |
| `game_date` | Date of game (YYYY-MM-DD) |
| `league` | League code (e.g. AHL) |
| `season` | Season year |
| `home_team` | Home team abbreviation |
| `away_team` | Away team abbreviation |
| `home_team_id` | Home team unique identifier |
| `away_team_id` | Away team unique identifier |
| `home_score` | Final home team score |
| `away_score` | Final away team score |
| `game_outcome` | Result: `home_win`, `away_win`, or `tie` |
| `home_start_net` | Whether home team starts at `pos_x` (89,0) or `neg_x` (-89,0); controls coordinate adjustment |

---

## events

One record per event.

| Field | Description |
|-------|-------------|
| `game_id` | Links to games |
| `period` | Period number |
| `period_time` | Time elapsed in period (seconds) |
| `game_stint` | Stint index; new stint when a player enters or leaves the ice |
| `sl_event_id` | Event index within game |
| `sequence_id` | Faceoff-to-whistle stretch; new sequence at each faceoff |
| `player_id` | Primary actor (faceoff taker, passer, etc.); may be null |
| `player_name` | Player name |
| `team` | Actor team abbreviation |
| `team_id` | Actor team identifier |
| `opp_team` | Opponent abbreviation |
| `opp_team_id` | Opponent team identifier |
| `event_type` | Type of event (see below) |
| `outcome` | `successful`, `failed`, or undetermined |
| `flags` | Additional metadata (e.g. zone, handedness) |
| `description` | Text description |
| `detail` | Subtype of event (see event_type details in instructions) |
| `sl_xg_all_shots` | Expected goals (when applicable) |
| `x`, `y` | Ice coordinates; x from -100 to 100, y from -42.5 to 42.5 |
| `x_adj`, `y_adj` | Coordinates adjusted so subject team's offensive zone is to the right |
| `has_tracking_data` | 1 if tracking exists for this event, 0 otherwise |
| `event_player_tracked` | 1 if the primary actor has tracking, 0 otherwise |

### event_type (and detail values)

- **faceoff**: recoveredwithentry, recoveredwithshotonnet, recoveredwithexit, recoveredwithshotonnetandslotshot, recoveredwithslotshot, recovered, none
- **lpr** (loose puck recovery): contested, faceoff, rebound, opdump, nofore, hipresopdump, none
- **shot**: outside, outsideblocked, slot, slotblocked
- **pass**: d2d, outlet, stretch, ozentrystretch, north, south, eastwest, ozentry, rush, slot, offboards, none
- **reception**: ozentry, regular
- **dumpout**: ice, boards, flip
- **dumpin**: dump, chip
- Others: deflection, failedpasslocation, receptionprevention, puckprotection, check, pressure, block, controlledentryagainst, carry, dumpinagainst, controlledbreakout, penaltydrawn, penalty, offside, icing, goal, whistle, assist, shootout, soshot, sogoal, socheck, socarry, solpr, sopuckprotection, penaltyshot, psshot, pscheck, pscarry, pslpr, pspuckprotection, teamwithozonright

---

## players

One record per player.

| Field | Description |
|-------|-------------|
| `player_id` | Unique player identifier |
| `player_name` | Display name (Last, First) |
| `last_name` | Last name |
| `first_name` | First name |
| `handed` | Stick handedness: L or R |
| `birth_date` | Date of birth |
| `position_group` | F (forward), D (defence), or G (goalie) |
| `primary_position` | C, LW, RW, D, G |

---

## stints

One record per player per stint.

| Field | Description |
|-------|-------------|
| `game_id` | Links to games |
| `period` | Period number |
| `period_time_start` | Start time of stint in period |
| `period_time_end` | End time of stint in period |
| `game_stint` | Stint index |
| `n_home_skaters` | Number of home skaters on ice |
| `n_away_skaters` | Number of away skaters on ice |
| `is_home_net_empty` | Home net empty (pulled goalie) |
| `is_away_net_empty` | Away net empty |
| `home_score` | Home score at stint start |
| `away_score` | Away score at stint start |
| `player_id` | Player on ice for this stint |
| `player_name` | Player name |
| `team_id` | Player's team |
| `team` | Team abbreviation |

---

## tracking

One record per player per event. Player positions at the moment of each event. Not every on-ice player has tracking (derived from broadcast video).

| Field | Description |
|-------|-------------|
| `game_id` | Links to games |
| `sl_event_id` | Links to events |
| `team_id` | Player's team |
| `team_name` | Team name |
| `player_id` | Player; may be null for unidentified objects |
| `player_name` | Player name |
| `tracking_x` | X position on ice |
| `tracking_y` | Y position on ice |
| `tracking_vel_x` | X velocity (if available) |
| `tracking_vel_y` | Y velocity (if available) |

---

## Processed: forechecks

Output of `scripts/01_forechecks.py`; written to `data/processed/forechecks.parquet`. One record per forecheck sequence. **Outcome rules (success, failure, dropped):** see README § Forecheck outcomes.

| Field | Description |
|-------|-------------|
| `fc_sequence_id` | Row index |
| `game_id` | Game |
| `period` | Period (at dump-in) |
| `period_time` | Period time at dump-in (seconds) |
| `sequence_id` | Event sequence (faceoff-to-whistle) |
| `sl_event_id_start` | Event id of first LPR under pressure (sequence start) |
| `sl_event_id_end` | Event id of terminal event |
| `pressing_team_id` | Team applying forecheck (Team A) |
| `defending_team_id` | Team in possession after dump-in (Team B) |
| `dumpin_detail` | dump / chip |
| `lpr_detail` | LPR subtype (opdump, hipresopdump, etc.) |
| `puck_x_at_start`, `puck_y_at_start` | Puck location at sequence start |
| `sign_negative` | True if puck in negative-x zone (used for flip) |
| `outcome` | `"success"` or `"failure"` |
| `y` | 1 = success, 0 = failure |
| `terminal_event_type` | Event that ended the sequence (e.g. zone_exit, lpr, possession, whistle, goal, penalty) |
