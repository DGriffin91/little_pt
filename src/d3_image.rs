use glam::{UVec3, Vec3};

pub struct Image {
    pub data: Vec<Vec3>,
    pub size: UVec3,
}

impl Image {
    pub fn trilinear_sample(&self, uvw: Vec3) -> Vec3 {
        let size_x = self.size.x as usize;
        let size_y = self.size.y as usize;
        let size_z = self.size.z as usize;
        let fsize = self.size.as_vec3();

        let uvw = uvw * fsize - 0.5;

        let x0 = (uvw.x.floor().max(0.0) as usize).min(size_x - 1);
        let y0 = (uvw.y.floor().max(0.0) as usize).min(size_y - 1);
        let z0 = (uvw.z.floor().max(0.0) as usize).min(size_z - 1);

        let x1 = (x0 + 1).min(size_x - 1);
        let y1 = (y0 + 1).min(size_y - 1);
        let z1 = (z0 + 1).min(size_z - 1);

        let xd = uvw.x - x0 as f32;
        let yd = uvw.y - y0 as f32;
        let zd = uvw.z - z0 as f32;

        let index = |x, y, z| x + y * size_x + z * size_x * size_y;

        let c00 = self.data[index(x0, y0, z0)].lerp(self.data[index(x1, y0, z0)], xd);
        let c01 = self.data[index(x0, y0, z1)].lerp(self.data[index(x1, y0, z1)], xd);
        let c10 = self.data[index(x0, y1, z0)].lerp(self.data[index(x1, y1, z0)], xd);
        let c11 = self.data[index(x0, y1, z1)].lerp(self.data[index(x1, y1, z1)], xd);

        let c0 = c00.lerp(c10, yd);
        let c1 = c01.lerp(c11, yd);

        let c = c0.lerp(c1, zd);

        c
    }
}
