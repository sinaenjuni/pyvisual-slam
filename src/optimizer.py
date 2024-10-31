import numpy as np
import g2o

class Optimizer:
    def __init__(self):
        self.verbose = True
        self.use_robust_kernel = True
        self.thHuber2D = np.sqrt(5.99)
        self.thHuber3D = np.sqrt(7.815)

    def BundleAdjustment(self, frames, map_points, n_iters=10):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        num_frame = len(frames)
        for key_frame in frames:
            vertex = g2o.VertexSE3Expmap() 
            vertex.set_id(key_frame.get_id())
            vertex.set_estimate(g2o.SE3Quat(key_frame.get_Rcw(), key_frame.get_tcw())) # R(3,3), t(3,1)
            vertex.set_fixed( True if key_frame.get_id()==0 else False)
            optimizer.add_vertex(vertex)


        for point_3d in map_points:
            point3d_id = point_3d.get_id() + num_frame
            vertex = g2o.VertexPointXYZ()
            vertex.set_id(point3d_id)
            vertex.set_estimate(point_3d.get_pos())
            vertex.set_marginalized(True)
            # vertex.set_fixed(fixed_points)
            optimizer.add_vertex(vertex)
            
            observation = point_3d.get_observation()
            for frame, idx_point_img in observation:
                pose_id = frame.get_id()
                point_img = frame.get_kps()[idx_point_img]
                point_octave = frame.get_octave()[idx_point_img]
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, optimizer.vertex(point3d_id))
                edge.set_vertex(1, optimizer.vertex(pose_id)) # #1 of vertex is pose information
                edge.set_measurement(point_img)
                edge.set_parameter_id(0, 0)

                invSigma2 = frame.get_invLevelSigma2(point_octave)
                edge.set_information(np.eye(2)*invSigma2)
                # edge.set_information(np.eye(2))
                if self.use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(self.thHuber2D))

                edge.fx = frame.get_fx()
                edge.fy = frame.get_fy()
                edge.cx = frame.get_cx()
                edge.cy = frame.get_cy()
                optimizer.add_edge(edge)

        optimizer.set_verbose(self.verbose)
        optimizer.initialize_optimization()
        optimizer.optimize(n_iters)
    
        for frame in frames:
            vertex = optimizer.vertex(frame.get_id()) 
            pose = vertex.estimate().matrix()
            frame.set_Tcw(pose)


        for map_point in map_points:
            vertex = optimizer.vertex(map_point.get_id() + num_frame)
            point = vertex.estimate()
            map_point.update_pos(point)
