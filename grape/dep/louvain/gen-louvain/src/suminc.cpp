// File: modularity.cpp
// -- quality functions (for Newman-Girvan Suminc criterion) source file
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


#include "suminc.h"

using namespace std;


Suminc::Suminc(Graph & gr):Quality(gr,"Newman-Girvan [Suminc]") {
  cerr << "  creat suminc..." << endl;
  n2c.resize(size);
  com_size.resize(size);

  in.resize(size);
  tot.resize(size);
  
  // initialization
  for (int i=0 ; i<size ; i++) {
    n2c[i] = i;
    com_size[i] = g.node_size[i];
    in[i]  = g.nb_selfloops(i);
    tot[i] = g.weighted_degree(i);
  }
}

Suminc::~Suminc() {
  in.clear();
  tot.clear();
}

// long double
// Suminc::quality() {
//   long double q  = 0.0L;
//   long double m2 = g.total_weight;

//   for (int i=0 ; i<size ; i++) {
//     if (tot[i] > 0.0L)
//       q += in[i] - (tot[i]*tot[i]) / m2;
//   }

//   q /= m2;

//   return q;
// }

long double
Suminc::quality() {
  long double q  = 0.0L;
  long double m2 = g.total_weight;

  for (int i=0 ; i<size ; i++) {
    if (tot[i] > 0.0L && in[i] > 0.0L){
      /* modularity */
      // q += in[i] - (tot[i]*tot[i]) / m2;

      long double totc = tot[i];
      long double inc = in[i];
      /* method1 */
      // long double tail = (1-beta) * (3*inc - 2*totc) / m2;
      /* method2 */
      long double tail = (1-beta) * (2*inc - (totc - inc)*(totc - inc))/m2/2;
      q += (inc - (totc*totc) / m2) / m2 * beta + tail;
    }
  }

  // q /= m2;

  return q;
}
