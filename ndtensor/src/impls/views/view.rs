/*
    Appellation: view <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::TensorView;
use nd::Dimension;

impl<'a, A, D> TensorView<'a, A, D>
where
    D: Dimension,
{
    pub fn reborrow<'b>(&'b self) -> TensorView<'b, A, D> {
        // crate::TensorView {
        //     id: self.id,
        //     ctx: self.ctx,
        //     data: self.data.reborrow(),
        //     op: self.op.reborrow(),
        // }
        unimplemented!()
    }
}
