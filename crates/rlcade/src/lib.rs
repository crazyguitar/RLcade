pub mod envs;

#[cfg(feature = "python")]
mod py;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use py::env::{PyNesSmbEnv, PyNesVecSmbEnv};
#[cfg(feature = "python")]
use py::nes::{PyNes, PyScreen};

#[cfg(feature = "python")]
#[pymodule]
pub fn nes(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScreen>()?;
    m.add_class::<PyNes>()?;
    m.add_class::<PyNesSmbEnv>()?;
    m.add_class::<PyNesVecSmbEnv>()?;
    m.add_function(wrap_pyfunction!(py::nes::play, m)?)?;
    Ok(())
}
