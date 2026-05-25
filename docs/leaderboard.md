# MESA Leaderboard and Submission Notes

The public repo does not yet host a live leaderboard service.

## Recommended submission model

For a future leaderboard, require:

- `schema v2` results only
- manifest-backed dataset metadata
- split label
- adapter and model disclosure
- result JSON artifact

## Minimum submission fields

- MESA version
- schema version
- dataset version
- split
- adapter
- model/backend details
- summary metrics

## Verification rules

- reject submissions without dataset metadata
- reject submissions that do not label public-dev vs hidden-test
- reject legacy-v1 composite-only submissions for official ranking
