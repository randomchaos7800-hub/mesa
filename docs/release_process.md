# MESA Release Process

This document defines the release process for benchmark data and benchmark code.

## Release scopes

MESA releases can change one or more of:

- benchmark code
- scorer behavior
- dataset schema
- dataset contents
- reporting contract

## Required release updates

Every benchmark-affecting release should update:

- [CHANGELOG.md](/home/dino/mesa-benchmark/CHANGELOG.md:1)
- the relevant dataset manifest
- tests that enforce new dataset or reporting expectations

## Dataset release checklist

- [ ] dataset validates against the active schema
- [ ] manifest matches dataset item count and task types
- [ ] review log is updated
- [ ] dataset version is bumped
- [ ] tests pass

## Code release checklist

- [ ] public API changes are documented
- [ ] scorer changes are documented
- [ ] reporting changes are documented
- [ ] tests pass

## Versioning guidance

- schema changes should increment schema version when backward compatibility breaks
- curated dataset changes should increment `dataset_version`
- benchmark-release labels can move independently from package version

## Release notes

Release notes should call out:

- benchmark contract changes
- new datasets or splits
- new required metadata
- changed scoring behavior
- changed reporting expectations
