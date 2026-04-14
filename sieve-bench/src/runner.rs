use anyhow::Result;

use crate::types::{Episode, Hit};

pub trait Runner {
    fn name(&self) -> &'static str;
    fn prepare_stable(&mut self, ep: &Episode) -> Result<()>;
    fn begin_fresh_arrival(&mut self, ep: &Episode) -> Result<()>;
    fn wait_for_steady_state(&mut self, _ep: &Episode) -> Result<()> {
        Ok(())
    }
    fn search_at_deadline(
        &mut self,
        ep: &Episode,
        deadline: std::time::Duration,
        k: usize,
    ) -> Result<Vec<Hit>>;
    fn cleanup(&mut self) -> Result<()>;
}
