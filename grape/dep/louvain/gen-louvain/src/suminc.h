// File: modularity.h
// -- quality functions (for Suminc criterion) header file
//-----------------------------------------------------------------------------
// Community detection
// Based on the article "Fast unfolding of community hierarchies in large networks"
// Copyright (C) 2008 V. Blondel, J.-L. Guillaume, R. Lambiotte, E. Lefebvre
//
// And based on the article
// Copyright (C) 2013 R. Campigotto, P. Conde Céspedes, J.-L. Guillaume
//
// This file is part of Louvain algorithm.
// 
// Louvain algorithm is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// Louvain algorithm is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with Louvain algorithm.  If not, see <http://www.gnu.org/licenses/>.
//-----------------------------------------------------------------------------
// Author   : E. Lefebvre, adapted by J.-L. Guillaume and R. Campigotto
// Email    : jean-loup.guillaume@lip6.fr
// Location : Paris, France
// Time	    : July 2013
//-----------------------------------------------------------------------------
// see README.txt for more details


#ifndef SUMINC_H
#define SUMINC_H

#include "quality.h"

using namespace std;


class Suminc: public Quality {
 public:

  vector<long double> in, tot; // used to compute the quality participation of each community

  Suminc(Graph & gr);
  ~Suminc();

  inline void remove(int node, int comm, long double dnodecomm);

  inline void insert(int node, int comm, long double dnodecomm);

  inline long double gain(int node, int comm, long double dnodecomm, long double w_degree);

  long double quality();
};


inline void
Suminc::remove(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);

  in[comm]  -= 2.0L*dnodecomm + g.nb_selfloops(node); // dnodecomm: K_i_in
  tot[comm] -= g.weighted_degree(node);
  
  com_size[comm] -= g.node_size[node];
  n2c[node] = -1;
}

inline void
Suminc::insert(int node, int comm, long double dnodecomm) {
  assert(node>=0 && node<size);
  
  in[comm]  += 2.0L*dnodecomm + g.nb_selfloops(node); // dnodecomm: K_i_in
  tot[comm] += g.weighted_degree(node);
  
  com_size[comm] += g.node_size[node];
  n2c[node] = comm;
}

inline long double
Suminc::gain(int node, int comm, long double dnc, long double degc) {
  assert(node>=0 && node<size);
  
  long double totc = tot[comm];
  long double m2   = g.total_weight;
  long double k_i_in = dnc;
  long double k_i = degc;
  long double inc = in[comm];
  long double tail = 0.0L;
  // if(inc > 0.0L){
  //   tail = 2*(beta - 1)*((totc + k_i)/(inc + 2*k_i_in) - (totc/inc));
  // }
  /* method1 */
  // tail = (1-beta) / m2 * 2 * (3*k_i_in - k_i);
  /* method2 */
  tail = (1-beta) / m2 / 2 * (4*k_i_in - (totc + k_i - inc - 2*k_i_in)*(totc + k_i - inc - 2*k_i_in) + (totc - inc)*(totc - inc));
  return (k_i_in - totc*k_i/m2)*beta/m2*2 + tail; // dnc: K_i_in, degc: K_i
  
  // long double totc = tot[comm];
  // long double m2   = g.total_weight; // 2*m
  // return (dnc - totc*degc/m2); // dnc: K_i_in, degc: K_i
}


#endif // SUMINC_H
