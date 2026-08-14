use std::sync::Arc;

/// Run a backend release operation only when storage owns the final tracked resource alias.
///
/// A queued cross-stream task can outlive its managed-memory binding while retaining a raw backend
/// resource. In that case explicit destruction is unsafe until the task has been submitted, so the
/// resource falls back to its ordinary drop behavior.
pub(crate) fn release_resource_before_drop_if_unaliased<T>(
    resource: T,
    resource_guard: Arc<()>,
    release: impl FnOnce(&T),
) -> bool {
    let released = Arc::strong_count(&resource_guard) == 1;
    if released {
        release(&resource);
    }
    released
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, sync::Arc};

    use super::release_resource_before_drop_if_unaliased;

    struct DropProbe<'a> {
        events: &'a RefCell<Vec<&'static str>>,
    }

    impl Drop for DropProbe<'_> {
        fn drop(&mut self) {
            self.events.borrow_mut().push("drop");
        }
    }

    #[test]
    fn alias_free_deallocated_resource_runs_release_before_drop_correctness() {
        let events = RefCell::new(Vec::new());
        let released = release_resource_before_drop_if_unaliased(
            DropProbe { events: &events },
            Arc::new(()),
            |probe| {
                probe.events.borrow_mut().push("release");
            },
        );

        assert!(released);
        assert_eq!(*events.borrow(), ["release", "drop"]);
    }

    #[test]
    fn aliased_deallocated_resource_defers_to_ordinary_drop_correctness() {
        let events = RefCell::new(Vec::new());
        let resource_guard = Arc::new(());
        let queued_alias = resource_guard.clone();
        let released = release_resource_before_drop_if_unaliased(
            DropProbe { events: &events },
            resource_guard,
            |probe| {
                probe.events.borrow_mut().push("release");
            },
        );

        assert!(!released);
        assert_eq!(*events.borrow(), ["drop"]);
        drop(queued_alias);
        assert_eq!(*events.borrow(), ["drop"]);
    }
}
