import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class SimpleRigidBodyRule(Rule):
    def __init__(self, tables=None, gravity=None, drag=0.01, ground_y=0.0, enable_collisions=True):
        super().__init__("Physics.SimpleRigidBody", priority=75)
        self.tables = list(tables) if tables else ["physics_body"]
        self.gravity = np.array(gravity if gravity is not None else [0.0, -9.8, 0.0], dtype=np.float32)
        self.drag = float(drag)
        self.ground_y = float(ground_y)
        self.enable_collisions = bool(enable_collisions)

    def evaluate(self, facts: FactBase):
        dt = facts.get_global("dt")
        try:
            dt = float(dt)
        except Exception:
            dt = 0.0
        if dt <= 0.0:
            return

        g = facts.get_global("gravity")
        if g is None:
            g = self.gravity
        else:
            try:
                g = np.array(g, dtype=np.float32).reshape(3)
            except Exception:
                g = self.gravity

        drag = facts.get_global("air_drag")
        try:
            drag = float(drag)
        except Exception:
            drag = self.drag
        if drag < 0.0:
            drag = 0.0

        ground_y = facts.get_global("ground_y")
        try:
            ground_y = float(ground_y)
        except Exception:
            ground_y = self.ground_y

        enable_collisions = facts.get_global("physics_enable_collisions")
        if enable_collisions is None:
            enable_collisions = self.enable_collisions
        enable_collisions = bool(enable_collisions)

        for table in self.tables:
            try:
                pos_x = facts.get_column(table, "pos_x")
                pos_y = facts.get_column(table, "pos_y")
                pos_z = facts.get_column(table, "pos_z")
                vel_x = facts.get_column(table, "vel_x")
                vel_y = facts.get_column(table, "vel_y")
                vel_z = facts.get_column(table, "vel_z")
            except KeyError:
                continue

            count = len(pos_x)
            if count <= 0:
                continue

            try:
                radius = facts.get_column(table, "radius")
            except KeyError:
                radius = None
            try:
                mass = facts.get_column(table, "mass")
            except KeyError:
                mass = None
            try:
                restitution = facts.get_column(table, "restitution")
            except KeyError:
                restitution = None

            if radius is None:
                radius = np.full(count, 0.25, dtype=np.float32)
            if mass is None:
                mass = np.full(count, 1.0, dtype=np.float32)
            if restitution is None:
                restitution = np.zeros(count, dtype=np.float32)

            dynamic = mass > 0.0
            if not np.any(dynamic):
                continue

            dvx = g[0] * dt
            dvy = g[1] * dt
            dvz = g[2] * dt

            vel_x = vel_x.copy()
            vel_y = vel_y.copy()
            vel_z = vel_z.copy()
            pos_x = pos_x.copy()
            pos_y = pos_y.copy()
            pos_z = pos_z.copy()

            vel_x[dynamic] += dvx
            vel_y[dynamic] += dvy
            vel_z[dynamic] += dvz

            if drag > 0.0:
                damp = float(np.exp(-drag * dt))
                vel_x[dynamic] *= damp
                vel_y[dynamic] *= damp
                vel_z[dynamic] *= damp

            pos_x[dynamic] += vel_x[dynamic] * dt
            pos_y[dynamic] += vel_y[dynamic] * dt
            pos_z[dynamic] += vel_z[dynamic] * dt

            floor_y = ground_y + radius
            hit = dynamic & (pos_y < floor_y)
            if np.any(hit):
                pos_y[hit] = floor_y[hit]
                vy = vel_y[hit]
                vy = np.where(vy < 0.0, -vy * restitution[hit], vy)
                vel_y[hit] = vy

            if enable_collisions and count > 1:
                self._resolve_sphere_collisions(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, radius, mass)

            facts.set_column(table, "pos_x", pos_x)
            facts.set_column(table, "pos_y", pos_y)
            facts.set_column(table, "pos_z", pos_z)
            facts.set_column(table, "vel_x", vel_x)
            facts.set_column(table, "vel_y", vel_y)
            facts.set_column(table, "vel_z", vel_z)

    def _resolve_sphere_collisions(self, px, py, pz, vx, vy, vz, radius, mass):
        max_r = float(np.max(radius)) if len(radius) else 0.25
        cell = max(0.25, max_r * 2.5)
        inv_cell = 1.0 / cell

        cx = np.floor(px * inv_cell).astype(np.int32)
        cz = np.floor(pz * inv_cell).astype(np.int32)

        buckets = {}
        for i in range(len(px)):
            if mass[i] <= 0.0:
                continue
            key = (int(cx[i]), int(cz[i]))
            lst = buckets.get(key)
            if lst is None:
                buckets[key] = [i]
            else:
                lst.append(i)

        if not buckets:
            return

        neigh = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        for (bx, bz), ids in list(buckets.items()):
            for dx, dz in neigh:
                other = buckets.get((bx + dx, bz + dz))
                if not other:
                    continue
                for i in ids:
                    for j in other:
                        if j <= i:
                            continue
                        dxp = float(px[j] - px[i])
                        dyp = float(py[j] - py[i])
                        dzp = float(pz[j] - pz[i])
                        rr = float(radius[i] + radius[j])
                        dist2 = dxp * dxp + dyp * dyp + dzp * dzp
                        if dist2 <= 0.0 or dist2 >= rr * rr:
                            continue
                        dist = float(np.sqrt(dist2))
                        nx = dxp / dist
                        ny = dyp / dist
                        nz = dzp / dist
                        pen = rr - dist

                        mi = float(mass[i])
                        mj = float(mass[j])
                        inv_mi = 0.0 if mi <= 0.0 else 1.0 / mi
                        inv_mj = 0.0 if mj <= 0.0 else 1.0 / mj
                        inv_sum = inv_mi + inv_mj
                        if inv_sum <= 0.0:
                            continue

                        si = pen * (inv_mi / inv_sum)
                        sj = pen * (inv_mj / inv_sum)
                        px[i] -= nx * si
                        py[i] -= ny * si
                        pz[i] -= nz * si
                        px[j] += nx * sj
                        py[j] += ny * sj
                        pz[j] += nz * sj

                        rvx = float(vx[j] - vx[i])
                        rvy = float(vy[j] - vy[i])
                        rvz = float(vz[j] - vz[i])
                        vn = rvx * nx + rvy * ny + rvz * nz
                        if vn >= 0.0:
                            continue

                        e = 0.0
                        j_imp = -(1.0 + e) * vn / inv_sum
                        ix = j_imp * nx
                        iy = j_imp * ny
                        iz = j_imp * nz
                        vx[i] -= ix * inv_mi
                        vy[i] -= iy * inv_mi
                        vz[i] -= iz * inv_mi
                        vx[j] += ix * inv_mj
                        vy[j] += iy * inv_mj
                        vz[j] += iz * inv_mj

