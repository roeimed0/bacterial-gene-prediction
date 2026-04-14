---
name: Refactor
about: Improve code structure, readability, or maintainability without changing behavior
title: "REF: <short description>"
labels: ["refactor", "needs-triage"]
assignees: ''
---

## What needs to be refactored?

<!-- File path(s) and the specific function, class, or module -->

## Why is this a problem?

<!-- Describe the issue: duplicated logic, poor naming, tight coupling, God function, etc. -->

## Proposed change

<!-- Describe the structural improvement — e.g., extract helper, split module, rename for clarity -->

## Scope

- [ ] No behavior change (pure refactor)
- [ ] Internal API change (affects other modules but not public interface)
- [ ] Public API change (requires version bump / migration note)

## Risks

<!-- Any areas where a refactor could accidentally break behavior — call out things that need careful testing -->

## Related issues / PRs

<!-- Link any bug, feature, or perf issue this refactor would unblock or simplify -->
