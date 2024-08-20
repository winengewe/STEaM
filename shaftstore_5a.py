import numpy as np

from scipy.integrate import odeint

class ShaftStore(object):

    def __init__(self,Rx, XTC, number_nodes,number_layers,RLA,RLW,deltaT):
        
        #self.Rx = np.array([1000.,500.,100.,50.,25.,15.,10.,6.,4.5,3.5,0.,0.,0.]) # Node outer radii (m)
        #self.Rx = np.array([256.,128.,64.,32.,16.,8.,4.,2.,1.,0.,-3.5,-3.5,-3.5]) + 3.5 # Node outer radii (m)
        #self.XTC = np.array([[2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3.,0.026],[2.5,2.5,2.5,2.5,2.5,2.5,2.5,2.5,3.,0.6]]) # Node thermal conductivity (W/mK)
        self.Rx = Rx # Node outer radii (m)
        self.XTC = XTC # Node thermal conductivity (W/mK)
        self.number_nodes = number_nodes
        self.number_layers = number_layers
        self.RLA = RLA # Air Layer (m)
        self.RLW = RLW # Water layers height (m)
        self.deltaT = deltaT # temperature difference between DH supply and return pipe
    
    def coefficient_base(self, layer, node):
        
        Rx = self.Rx
        XTC = self.XTC
        if layer == 0:
            RL = self.RLA
        else:
            RL = self.RLW
        
        if layer > 0:
            layer = 1 # Temp!!! 
        
        if node == 0: # Outer Ring
            rf2 = 0.5 * (Rx[node] + Rx[node+1])
            rf3 = 0.5 * (Rx[node+1] + Rx[node+2])
            RTsx = np.log(rf2 / rf3)
            RTs1a = np.log(Rx[node] / rf2)
            Cxy = ((2. * np.pi * XTC[layer,node] * RL) / RTs1a)
            Cyz = ((2. * np.pi * XTC[layer,node] * RL) / RTsx)
        elif node == self.number_nodes-3: # Soil Inner Layer
            rf1 = 0.5 * (Rx[node-1] + Rx[node])
            rf2 = 0.5 * (Rx[node] + Rx[node+1])
            rf3 = 0.5 * (Rx[node+1] + Rx[node+2])
            RTsx = np.log(rf1 / rf2)
            RTsy = np.log(rf2 / Rx[node+1])
            RTsz = np.log(Rx[node+1] / rf3)
            Cxy = ((2. * np.pi * XTC[layer,node] * RL) / RTsx)
            Cyz = ((2. * np.pi * RL) / ((RTsy / XTC[layer,node]) + (RTsz / XTC[layer,node+1])))
        elif node == self.number_nodes-2: # Concrete Wall
            rf1 = 0.5 * (Rx[node-1] + Rx[node])
            rf2 = 0.5 * (Rx[node] + Rx[node+1])
            rf3 = 0.5 * (Rx[node+1] + Rx[node+2])
            RTsx = np.log(rf1 / Rx[node])
            RTsy = np.log(Rx[node] / rf2)
            RTsz = np.log(rf2 / Rx[node+1])
            Cxy = ((2. * np.pi * RL) / ((RTsx / XTC[layer,node-1]) + (RTsy / XTC[layer,node])))
            Cyz = ((2. * np.pi * RL) / ((RTsy / XTC[layer,node]) + (RTsz / XTC[layer,node+1])))
        elif node == self.number_nodes-1: # Fluid
            rf1 = 0.5 * (Rx[node-1] + Rx[node])
            RTsx = np.log(Rx[node-1] / rf1)
            RTsy = np.log(rf1 / Rx[node])
            Cxy = ((2. * np.pi * RL) / ((RTsx / XTC[layer,node-1]) + (RTsy / XTC[layer,node])))
            Cyz = 0.
        elif node > 0 and node < self.number_nodes-3: # Intermediate Soil Rings
            rf1 = 0.5 * (Rx[node-1] + Rx[node])
            rf2 = 0.5 * (Rx[node] + Rx[node+1])
            rf3 = 0.5 * (Rx[node+1] + Rx[node+2])
            RTsx = np.log(rf1 / rf2)
            RTsy = np.log(rf2 / rf3)
            Cxy = ((2. * np.pi * XTC[layer,node] * RL) / RTsx)
            Cyz = ((2. * np.pi * XTC[layer,node] * RL) / RTsy)
        else:
            Cxy = 0.
            Cyz = 0.
            
        return Cxy, Cyz
    
    def coefficient_A(self, layer, node, Cxy, Cyz, mdotu, mdotd):
    
        if node < self.number_nodes - 1:
            cp = 800. # J/kg Soil Heat Capacity
            Por = 0.2 # Soil porosity
            Den = (1750. * (1. - Por)) + (1000. * Por) # Node density
        else:
            if layer == 0:
                cp = 1005.
                Den = 1.23
            else:
                cp = 4184.
                Den = 1000.
                
        if layer == 0:
            RL = self.RLA
        else:
            RL = self.RLW
        
        node_mass = RL * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
    
        if node < self.number_nodes-1 or layer == 0:
            A =  (-Cxy - Cyz) / (node_mass * cp)
        else:
            A = (-Cxy - Cyz - (mdotu + mdotd) * cp) / (node_mass * cp)
            
        return A
    
    
    def coefficient_B(self, layer, node, Cxy):
        
        if node < self.number_nodes - 1:
            cp = 800. # J/kg Soil Heat Capacity
            Por = 0.2 # Soil porosity
            Den = (1750. * (1. - Por)) + (1000. * Por) # Node density
        else:
            if layer == 0:
                cp = 1005.
                Den = 1.23
            else:
                cp = 4184.
                Den = 1000.
                
        if layer == 0:
            RL = self.RLA
        else:
            RL = self.RLW
        
        node_mass = RL * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
            
        B = Cxy / (node_mass * cp)
        
        return B
    
    def coefficient_C(self, layer, node, Cyz):
        
        if node < self.number_nodes - 1:
            cp = 800. # J/kg Soil Heat Capacity
            Por = 0.2 # Soil porosity
            Den = (1750. * (1. - Por)) + (1000. * Por) # Node density
        else:
            if layer == 0:
                cp = 1005.
                Den = 1.23
            else:
                cp = 4184.
                Den = 1000.
                
        if layer == 0:
            RL = self.RLA
        else:
            RL = self.RLW
        
        node_mass = RL * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
        
        C = Cyz / (node_mass * cp)
        
        return C
    
    def coefficient_D(self, layer, node, mdotd):
    
        if node == self.number_nodes - 1:
            cp = 4184.
            Den = 1000.
            
            node_mass = self.RLW * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
        
            if layer == 0:
                D = 0
            else:
                D = (mdotd * cp) / (node_mass * cp)
                
            # print(node, node_mass, mdotd)
                
        else:
            D = 0
                
        return D
    
    def coefficient_E(self, layer, node, mdotu):
    
        if node == self.number_nodes - 1:
            cp = 4184.
            Den = 1000.
            
            node_mass = self.RLW * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
        
            if layer == 0:
                E = 0
            else:
                E = (mdotu * cp) / (node_mass * cp)
                
        else:
            E = 0
            
        return E
    
    def coefficient_F(self, layer, node):
        
        if layer == 0:
            RL = self.RLA
        else:
            RL = self.RLW
        
        if layer > 0:
            layer = 1 # Temp!!!
        
        cp = 800. # J/kg Soil Heat Capacity
        Por = 0.2 # Soil porosity
        Den = (1750. * (1. - Por)) + (1000. * Por) # Node density
        
        node_mass = RL * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
    
        if node == 0:
            rf2 = 0.5 * (self.Rx[node] + self.Rx[node+1])
            RTsx = np.log(self.Rx[node] / rf2)
            Cxy = ((2. * np.pi * self.XTC[layer,node] * RL) / RTsx)
            
            F = (10. * Cxy) / (node_mass * cp)
        else:
            F = 0
    
        return F
    
    def coefficient_Z(self, layer, node, mdotu, mdotd, store_temp, return_temp):
        
        if node == self.number_nodes - 1:
            cp = 4184.
            Den = 1000.
            
            node_mass = self.RLW * Den * np.pi * (self.Rx[node]**2 - self.Rx[node+1]**2)
        
            if layer == 0:
                Z = 0
            elif layer == 1:
                Z = (mdotd * cp * store_temp) / (node_mass * cp)
            elif layer == self.number_layers - 1:
                Z = (mdotu * cp * return_temp) / (node_mass * cp)
            else:
                Z = 0
                
        else:
            Z = 0
            
        return Z
    
    def set_of_coefficients(self, nodes_temp, mdotu, mdotd, store_temp, return_temp):
    
        c = [[] for i in range(self.number_layers)]
        
        for layer in range(self.number_layers):
            
            for node in range(self.number_nodes):
                
                Cxy, Cyz = self.coefficient_base(layer, node)
                
                coefficients = {'A': self.coefficient_A(layer, node, Cxy, Cyz, mdotu, mdotd),
                                'B': self.coefficient_B(layer, node, Cxy),
                                'C': self.coefficient_C(layer, node, Cyz),
                                'D': self.coefficient_D(layer, node, mdotd),
                                'E': self.coefficient_E(layer, node, mdotu),
                                'F': self.coefficient_F(layer, node),
                                'Z': self.coefficient_Z(layer, node, mdotu, mdotd, store_temp, return_temp),
                                }
                
                c[layer].append(coefficients)
        
        return c
    
    def new_nodes_temp(self, nodes_temp, store_temp, return_temp, charge, discharge, MTS, deltaT):
    
        def model_temp(z, t, c):
            dzdt = []
            
            for layer in range(self.number_layers):
                            
                for lnode in range(self.number_nodes):
        
                    node = (layer * self.number_nodes) + lnode            
        
                    if lnode == 0: # Furthest earth 'ring'
                        Ti = nodes_temp[node]
                        Ti_b = nodes_temp[node + 1]
        
                        dTdt = MTS * (c[layer][lnode]['A'] * Ti +
                                c[layer][lnode]['C'] * Ti_b +
                                c[layer][lnode]['F'])
        
                        dzdt.append(dTdt)
        
                    elif lnode == (self.number_nodes - 1): # Store Fluid
                    
                        if layer == 1: # Top Water Layer
                            Ti = nodes_temp[node]
                            Ti_a = nodes_temp[node - 1]
                            Td = nodes_temp[node + self.number_nodes]
            
                            dTdt = MTS * (c[layer][lnode]['A'] * Ti +
                                          c[layer][lnode]['B'] * Ti_a + 
                                          c[layer][lnode]['E'] * Td +
                                          c[layer][lnode]['Z']
                                          ) 
        
                            dzdt.append(dTdt)
                        
                        elif layer == self.number_layers - 1: # Bottom Water Layer
                            Ti = nodes_temp[node]
                            Ti_a = nodes_temp[node - 1]
                            Tu = nodes_temp[node - self.number_nodes]
            
                            dTdt = MTS * (c[layer][lnode]['A'] * Ti +
                                          c[layer][lnode]['B'] * Ti_a + 
                                          c[layer][lnode]['D'] * Tu +
                                          c[layer][lnode]['Z']
                                          ) 
                            
                            dzdt.append(dTdt)
                            
                        else:
                            Ti = nodes_temp[node]
                            Ti_a = nodes_temp[node - 1]
                            Tu = nodes_temp[node - self.number_nodes]
                            Td = nodes_temp[node + self.number_nodes]
            
                            dTdt = MTS * (c[layer][lnode]['A'] * Ti +
                                          c[layer][lnode]['B'] * Ti_a + 
                                          c[layer][lnode]['D'] * Tu +
                                          c[layer][lnode]['E'] * Td
                                          ) 
            
                            dzdt.append(dTdt)
        
                    else:
                        Ti = nodes_temp[node]
                        Ti_b = nodes_temp[node + 1]
                        Ti_a = nodes_temp[node - 1]
        
                        dTdt = MTS * (c[layer][lnode]['A'] * Ti +
                                c[layer][lnode]['B'] * Ti_a +
                                c[layer][lnode]['C'] * Ti_b +
                                c[layer][lnode]['F'])
        
                        dzdt.append(dTdt)
    
            return dzdt
    
        node_temp_list = []
        node_temp_list.append(nodes_temp)
        
        if charge > 0:
            mdotd = (charge * 3600.) / (MTS * 4.181 * (store_temp - nodes_temp[109])) # charge in kWh to kg/s, deltaT based on tank bottom temperature
        else:
            mdotd = 0.
            
        if discharge > 0:
            mdotu = (discharge * 3600.) / (MTS * 4.181 * deltaT) # discharge in kWh to kg/s, 10C deltaT
        else:
            mdotu = 0.    
            
        #print(mdotu, discharge, mdotd, charge, (store_temp - nodes_temp[109]))
    
        # solve ODE
        for i in range(1, 2):
            # span for next time step
            tspan = [i - 1, i]
            # solve for next step
            
            # new coefficients
            coefficients = self.set_of_coefficients(nodes_temp, mdotu, mdotd, store_temp, return_temp)
            
            z = odeint(
                model_temp, nodes_temp, tspan,
                args=(coefficients,))
    
            nodes_temp = z[1]
            int_nt = nodes_temp[19:110:10]
            int_nt1 = sorted(int_nt, reverse=True)
            nodes_temp[19:110:10] = int_nt1
            node_temp_list.append(nodes_temp)
            
        # Calculate heat losses
        Hloss = 0.
        for layer in range(1,self.number_layers):
            rf1 = 0.5 * (self.Rx[8] + self.Rx[9])
            RTsx = np.log(self.Rx[8] / rf1)
            RTsy = np.log(rf1 / self.Rx[9])
            Cxy = ((2. * np.pi * self.RLW) / ((RTsx / self.XTC[1,8]) + (RTsy / self.XTC[1,9])))
            
            Hloss += Cxy * (nodes_temp[layer * self.number_nodes + 9] - nodes_temp[layer * self.number_nodes + 8]) * (MTS/3600.) * 0.001 # kWh
    
        return node_temp_list, Hloss