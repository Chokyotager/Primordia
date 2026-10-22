//! Minimal columnar table buffered in memory and streamed to Parquet.

use anyhow::Result;
use arrow_array::{ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum V {
    I(i64),
    F(f64),
    S(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ty {
    I,
    F,
    S,
}

pub struct Table {
    pub name: String,
    pub cols: Vec<(String, Ty)>,
    rows: Vec<Vec<V>>,
    writer: Option<ArrowWriter<std::fs::File>>,
    schema: Arc<Schema>,
    pub total_rows: u64,
}

impl Table {
    pub fn new(dir: Option<&Path>, name: &str, cols: &[(&str, Ty)]) -> Result<Table> {
        let fields: Vec<Field> = cols
            .iter()
            .map(|(n, t)| {
                Field::new(
                    *n,
                    match t {
                        Ty::I => DataType::Int64,
                        Ty::F => DataType::Float64,
                        Ty::S => DataType::Utf8,
                    },
                    false,
                )
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let writer = match dir {
            Some(d) => {
                let f = std::fs::File::create(d.join(format!("{name}.parquet")))?;
                let props = WriterProperties::builder().set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?)).build();
                Some(ArrowWriter::try_new(f, schema.clone(), Some(props))?)
            }
            None => None,
        };
        Ok(Table { name: name.into(), cols: cols.iter().map(|(n, t)| (n.to_string(), *t)).collect(), rows: vec![], writer, schema, total_rows: 0 })
    }

    pub fn push(&mut self, row: Vec<V>) -> Result<()> {
        debug_assert_eq!(row.len(), self.cols.len());
        self.rows.push(row);
        self.total_rows += 1;
        if self.rows.len() >= 4096 {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let Some(w) = self.writer.as_mut() else {
            self.rows.clear();
            return Ok(());
        };
        let mut arrays: Vec<ArrayRef> = vec![];
        for (j, (_, t)) in self.cols.iter().enumerate() {
            let a: ArrayRef = match t {
                Ty::I => Arc::new(Int64Array::from(self.rows.iter().map(|r| if let V::I(x) = r[j] { x } else { 0 }).collect::<Vec<_>>())),
                Ty::F => Arc::new(Float64Array::from(
                    self.rows
                        .iter()
                        .map(|r| match r[j] {
                            V::F(x) => x,
                            V::I(x) => x as f64,
                            _ => f64::NAN,
                        })
                        .collect::<Vec<_>>(),
                )),
                Ty::S => Arc::new(StringArray::from(self.rows.iter().map(|r| if let V::S(x) = &r[j] { x.clone() } else { String::new() }).collect::<Vec<_>>())),
            };
            arrays.push(a);
        }
        let batch = RecordBatch::try_new(self.schema.clone(), arrays)?;
        w.write(&batch)?;
        self.rows.clear();
        Ok(())
    }

    pub fn close(&mut self) -> Result<()> {
        self.flush()?;
        if let Some(w) = self.writer.take() {
            w.close()?;
        }
        Ok(())
    }
}
