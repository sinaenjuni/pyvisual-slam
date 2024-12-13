import numpy as np
import g2o
# from map import Map

class Optimizer:
    def __init__(self):
        self.verbose = True
        self.use_robust_kernel = True
        self.thHuber2D = np.sqrt(5.99)
        self.thHuber3D = np.sqrt(7.815)

    def BundleAdjustment(self, map, n_iters=10):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        max_frame_id = 0
        for key_frame in map.key_frames:
            vertex = g2o.VertexSE3Expmap() 
            id = key_frame.id
            vertex.set_id(id)
            vertex.set_estimate(g2o.SE3Quat(key_frame.Rcw, key_frame.tcw)) # R(3,3), t(3,1)
            vertex.set_fixed( True if key_frame.id==0 else False)
            optimizer.add_vertex(vertex)
            max_frame_id = max(max_frame_id, id)

        for point_3d in map.points3d:
            point3d_id = point_3d.get_id() + max_frame_id + 1
            vertex = g2o.VertexPointXYZ()
            vertex.set_id(point3d_id)
            vertex.set_estimate(point_3d.get_pos())
            vertex.set_marginalized(True)
            # vertex.set_fixed(fixed_points)
            optimizer.add_vertex(vertex)
            
            observation = point_3d.get_observation()
            for frame, idx_point_img in observation:
                pose_id = frame.id
                point_img = frame.kps[idx_point_img]
                # point_octave = frame.octave[idx_point_img]
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, optimizer.vertex(point3d_id))
                edge.set_vertex(1, optimizer.vertex(pose_id)) # #1 of vertex is pose information
                edge.set_measurement(point_img)
                edge.set_parameter_id(0, 0)

                # invSigma2 = frame.sigma2inv(point_octave)
                invSigma2 = frame.scale2inv[idx_point_img]
                edge.set_information(np.eye(2) * invSigma2)
                # edge.set_information(np.eye(2))
                if self.use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(self.thHuber2D))

                edge.fx = frame.camera.fx
                edge.fy = frame.camera.fy
                edge.cx = frame.camera.cx
                edge.cy = frame.camera.cy
                optimizer.add_edge(edge)

        optimizer.set_verbose(self.verbose)
        optimizer.initialize_optimization()
        optimizer.optimize(n_iters)
    
        for frame in map.key_frames:
            vertex = optimizer.vertex(frame.id) 
            pose = vertex.estimate().matrix()
            frame.set_Tcw(pose)

        for map_point in map.points3d:
            vertex = optimizer.vertex(map_point.get_id() + max_frame_id + 1)
            point = vertex.estimate()
            map_point.update_pos(point)
    
    def pose_optimization(self, frame, n_iters=10):
        n_corr_points = len(frame.points3d)
        n_bad_points = 0
        if n_corr_points < 3:
            print(f'''[Fail, pose_optimization]: The number of corresponding 
                    points ({len(frame.points3d)}) is lacking.''')
            return False
        
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        vertex = g2o.VertexSE3Expmap() 
        vertex.set_id(0)
        vertex.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw)) # R(3,3), t(3,1)
        vertex.set_fixed(False)
        optimizer.add_vertex(vertex)

        edges = []
        for img_point_idx, point3d in frame.points3d.items():
            # add edge
            edge = None 

            #print('adding mono edge between point ', p.id,' and frame ', frame.id)
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()

            edge.set_vertex(0, optimizer.vertex(0))
            edge.set_measurement(frame.kps[img_point_idx]) # corrispoding image point

            invSigma2 = frame.scale2inv[img_point_idx]
            edge.set_information(np.eye(2)*invSigma2)
            edge.set_robust_kernel(g2o.RobustKernelHuber(self.thHuber2D))

            edge.fx = frame.camera.fx
            edge.fy = frame.camera.fy
            edge.cx = frame.camera.cx
            edge.cy = frame.camera.cy
            edge.Xw = point3d.pos[0:3] # corrisponding 3d point
            
            optimizer.add_edge(edge)
            edges.append((img_point_idx, edge))

        for round in range(4):
            optimizer.set_verbose(self.verbose)
            optimizer.initialize_optimization()
            optimizer.optimize(n_iters)

            if round == 2: # prevent overftting ?
                for img_point_idx, edge in edges:
                    edge.set_robust_kernel(None)

            for img_point_idx, edge in edges:
                edge.compute_error()
                chi2 = edge.chi2()
                # https://github.com/RainerKuemmerle/g2o/issues/259#issuecomment-367556043
                edge.set_level(1 if chi2 > self.thHuber2D else 0)

            n_bad_points = len([False for e in edges if e[1].level()==1])
            n_good_points = n_corr_points - n_bad_points
            if n_good_points < 10:
                print(f'''[Fail, pose_optimization]: 
                        Not enough edges({n_good_points})''')
                return False
        
        print(f'''[Pass, pose_optimization]: n_edge({len(optimizer.edges())})
                                            error({optimizer.active_chi2() / max(1, n_good_points):.4f})''')
        vertex = optimizer.vertex(0) 
        pose = vertex.estimate().matrix()
        frame.set_Tcw(pose)
        return True