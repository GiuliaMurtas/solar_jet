//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file jet.cpp
//  \brief Problem generator for the solar jet problem.  Works in Cartesian coordinates.
//
// REFERENCE:

// C++ headers
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../globals.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../utils/utils.hpp"

#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {

// Initial pressure

Real pres_init(const Real bx, const Real by, const Real bz, const Real B_polar,
    const int pres_balance, const int pert_B, const Real b0, const Real Bguide,
    const Real phi_pert, const Real beta0, const Real lx, const Real xca,
    const Real yca, const Real zca, const Real xcb, const Real ycb, const Real zcb);
}

// Boundary conditions
void SymmInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);
void OpenInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int is, int ie, int js, int je, int ks, int ke, int ngh);
void OpenOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);
void OpenInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);
void OpenOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReduceInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);
void ReduceOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                   FaceField &b, Real time, Real dt,
                   int is, int ie, int js, int je, int ks, int ke, int ngh);

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief solar jet problem generator
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Parameters in initial configurations
  Real beta0, B_polar;
  Real b0 = 1.0, Bguide; // Normalized magnetic field
  Real phi_pert, vin_pert;
  Real xmin, xmax, lx; // Grid dimensions
  Real ymin, ymax, ly;
  Real gamma_adi, gamma_adi_red, gm1, iso_cs;
  Real xca, yca, zca, xcb, ycb, zcb;
  Real rho, pgas;

  int pres_balance = 1; // In default, the initial pressure is balanced
  int uniform_rho = 0;  // In default, the initial density is not uniform
  int pert_B = 1;       // In default, magnetic field is perturbed
  int pert_V = 0;       // In default, velocity field is not perturbed
  int random_vpert = 1; // In default, random velocity perturbation is used

  xmin  = pin->GetReal("mesh", "x1min");
  xmax  = pin->GetReal("mesh", "x1max");
  ymin  = pin->GetReal("mesh", "x2min");
  ymax  = pin->GetReal("mesh", "x2max");

  beta0 = pin->GetReal("problem", "beta0");
  vin_pert = pin->GetReal("problem", "vin_pert");
  random_vpert = pin->GetInteger("problem", "random_vpert");
  Bguide   = pin->GetReal("problem", "Bguide");
  phi_pert = pin->GetReal("problem", "phi_pert");
  pres_balance = pin->GetInteger("problem", "pres_balance");
  uniform_rho = pin->GetInteger("problem", "uniform_rho");
  pert_B = pin->GetInteger("problem", "pert_B");
  pert_V = pin->GetInteger("problem", "pert_V");
  b0 = pin->GetReal("problem","b0");

  B_polar = pin->GetReal("problem","B_polar");
  xca = pin->GetReal("problem","xca");
  yca = pin->GetReal("problem","yca");
  zca = pin->GetReal("problem","zca");
  xcb = pin->GetReal("problem","xcb");
  ycb = pin->GetReal("problem","ycb");
  zcb = pin->GetReal("problem","zcb");

  // initialize global variables
  if (NON_BAROTROPIC_EOS) {
    gamma_adi = peos->GetGamma();
    gamma_adi_red = gamma_adi / (gamma_adi - 1.0);
    gm1 = (gamma_adi - 1.0);
  } else {
    iso_cs = pin->GetReal("hydro", "iso_sound_speed");
  }

  AthenaArray<Real> b1i_pert, b2i_pert, b1c_pert, b2c_pert;
  int nx1 = block_size.nx1 + 2*NGHOST;
  int nx2 = block_size.nx2 + 2*NGHOST;
  int nx3 = block_size.nx3 + 2*NGHOST;
  int level=loc.level;

  // Initialize interface fields
  
  // Compute cell-centered fields
  pfield->CalculateCellCenteredField(pfield->b, pfield->bcc, pcoord, is, ie, js, je, ks, ke);
  
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke+1, je+1, ie+1);
  for (int k=ks; k<=ke; k++) {
   for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {

      // Set primitives
      phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
      phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
      phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = 0.0;
      phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = 0.0;
      phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = 0.0;

      Real x1f = pcoord->x1f(i);
      Real x2f = pcoord->x2f(j);
      Real x3f = pcoord->x3f(k);
      
      Real x0 = x1f + xca;
      Real y0 = x2f + yca;
      Real z0 = x3f + zca;

      Real xc0 = x1f + xcb;
      Real yc0 = x2f + ycb;
      Real zc0 = x3f + zcb;

      pfield->b.x1f(k,j,i) = ((x0/std::pow((std::pow(x0,2.0)+std::pow(y0,2.0)+std::pow(z0,2.0)),1.5)*B_polar)-(xc0/std::pow((std::pow(xc0,2.0)+std::pow(yc0,2.0)+std::pow(zc0,2.0)),1.5)*B_polar*1.5))/b0;
      pfield->b.x2f(k,j,i) = ((y0/std::pow((std::pow(x0,2.0)+std::pow(y0,2.0)+std::pow(z0,2.0)),1.5)*B_polar)-(yc0/std::pow((std::pow(xc0,2.0)+std::pow(yc0,2.0)+std::pow(zc0,2.0)),1.5)*B_polar*1.5))/b0;
      pfield->b.x3f(k,j,i) = ((0.001+z0/std::pow((std::pow(x0,2.0)+std::pow(y0,2.0)+std::pow(z0,2.0)),1.5)*B_polar)-(zc0/std::pow((std::pow(xc0,2.0)+std::pow(yc0,2.0)+std::pow(zc0,2.0)),1.5)*B_polar*1.5))/b0;
        
      if(pfield->b.x1f(k,j,i)!=0.0) std::cout << pfield->b.x1f(k,j,i) << std::endl;

      // Set magnetic fields
      bb(IB1,k,j,i) = pfield->b.x1f(k,j,i);
      bb(IB2,k,j,i) = pfield->b.x2f(k,j,i);
      bb(IB3,k,j,i) = pfield->b.x3f(k,j,i);
    }
   }
  }
  // Initialize hydro variables
#ifdef RELATIVISTIC_DYNAMICS
  //peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);
  //bb.DeleteAthenaArray();
#else
  Real pgas_nr;
  int64_t iseed = -1 - gid; // Ensure a different initial random seed for each meshblock.
  for (int k=ks; k<=ke; k++) {
  for (int j=js; j<=je; j++) {
    for (int i=is; i<=ie; i++) {
      pgas_nr = pres_init(pfield->bcc(IB1,k,j,i), pfield->bcc(IB2,k,j,i),
          pfield->bcc(IB3,k,j,i), pres_balance, pert_B, b0, Bguide,
          phi_pert, beta0;
      // density
      if (uniform_rho)
        phydro->u(IDN,k,j,i) = b0 * b0;
      else
        phydro->u(IDN,k,j,i) = pgas_nr / (0.5 * beta0);
      // momentum
      if (pert_V) {
        if (random_vpert) {
          phydro->u(IM1,k,j,i) = vin_pert * (2.0*ran2(&iseed) - 1.0);
          phydro->u(IM2,k,j,i) = vin_pert * (2.0*ran2(&iseed) - 1.0);
        }
      } else {
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
      }
      phydro->u(IM1,k,j,i) *= phydro->u(IDN,k,j,i);
      phydro->u(IM2,k,j,i) *= phydro->u(IDN,k,j,i);
      phydro->u(IM3,k,j,i) = 0.0;

      if (NON_BAROTROPIC_EOS) {
        phydro->u(IEN,k,j,i) = pgas_nr/gm1;
        phydro->u(IEN,k,j,i) += 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                     SQR(phydro->u(IM2,k,j,i)) +
                                     SQR(phydro->u(IM3,k,j,i))) / phydro->u(IDN,k,j,i);
        phydro->u(IEN,k,j,i) += 0.5*(SQR(pfield->bcc(IB1,k,j,i)) +
                                     SQR(pfield->bcc(IB2,k,j,i)) +
                                     SQR(pfield->bcc(IB3,k,j,i)));
      }
    }
  }}
#endif

  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//  \brief
//========================================================================================

void Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  return;
}

int RefinementCondition(MeshBlock *pmb);

void Mesh::InitUserMeshData(ParameterInput *pin)
{
  // Enroll boundary value function pointers
  if (mesh_bcs[BoundaryFace::inner_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x1, OpenInnerX1);
  }
  if (mesh_bcs[BoundaryFace::outer_x1] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x1, OpenOuterX1);
  }
  if (mesh_bcs[BoundaryFace::inner_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, OpenInnerX2);
  }
  if (mesh_bcs[BoundaryFace::outer_x2] == GetBoundaryFlag("user")) {
    EnrollUserBoundaryFunction(BoundaryFace::outer_x2, OpenOuterX2);
  }
  if(adaptive==true)
    EnrollUserRefinementCondition(RefinementCondition);

  return;
}

// refinement condition: density and pressure curvature
int RefinementCondition(MeshBlock *pmb)
{
  AthenaArray<Real> &w = pmb->phydro->w;
  Real maxeps=0.0;
  for(int k=pmb->ks; k<=pmb->ke; k++) {
    for(int j=pmb->js; j<=pmb->je; j++) {
      for(int i=pmb->is; i<=pmb->ie; i++) {
        Real epsr= (std::abs(w(IDN,k,j,i+1)-2.0*w(IDN,k,j,i)+w(IDN,k,j,i-1))
                   +std::abs(w(IDN,k,j+1,i)-2.0*w(IDN,k,j,i)+w(IDN,k,j-1,i))
                   +std::abs(w(IDN,k+1,j,i)-2.0*w(IDN,k,j,i)+w(IDN,k-1,j,i)))/w(IDN,k,j,i);
        Real epsp= (std::abs(w(IPR,k,j,i+1)-2.0*w(IPR,k,j,i)+w(IPR,k,j,i-1))
                   +std::abs(w(IPR,k,j+1,i)-2.0*w(IPR,k,j,i)+w(IPR,k,j-1,i))
                   +std::abs(w(IPR,k+1,j,i)-2.0*w(IPR,k,j,i)+w(IPR,k-1,j,i)))/w(IPR,k,j,i);
        Real eps = std::max(epsr, epsp);
        maxeps = std::max(maxeps, eps);
      }
    }
  }
  // refine : curvature > 10.0
  if(maxeps > 100.0) return 1;
  // derefinement: curvature < 5.0
  if(maxeps < 10.0) return -1;
  // otherwise, stay
  return 0;
}

namespace {
//========================================================================================
//  \brief initial gas pressure
//========================================================================================
Real pres_init(const Real bx, const Real by, const Real bz,
    const int pres_balance, const int pert_B, const Real b0, const Real Bguide,
    const Real phi_pert, const Real beta0, const Real lx)
{
  Real p0, pB, pmag_max;
  if (pres_balance && pert_B) {
    pmag_max = 0.5 * b0 * b0 * (1.0+phi_pert*PI/lx)*(1.0+phi_pert*PI/lx) + 0.5 * Bguide * Bguide;
    pB = 0.5 * (bx*bx + by*by + bz*bz);
    p0 = (beta0 * pmag_max + pmag_max) - pB;
  } else {
    p0 = beta0 * (0.5 * b0 * b0);
  }
  return p0;
}

} // namespace

//==============================================================================
// SymmInnerX1 boundary condition
//==============================================================================
void SymmInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(n,k,j,is-i) = prim(n,k,j,is+i-1);
      }
    }}
  }

  // Set velocity Vx
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        prim(IVX,k,j,is-i) = -prim(IVX,k,j,is+i-1);
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = b.x1f(k,j,is+i);
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = -b.x2f(k,j,is+i-1);
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,is+i-1);
      }
    }}
  }
}

//==============================================================================
// Open boundary condition at the left edge (inner x1 boundary)
//==============================================================================
void OpenInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                 FaceField &b, Real time, Real dt,
                 int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,is-i) = prim(n,k,j,is+i-1);
        }
      }
    }
  }

  // inflow restriction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        if (prim(IVX,k,j,is-i) > 0.0) {
          prim(IVX,k,j,is-i) = 0.0;
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(is-i)) = 2.0*b.x2f(k,j,(is-i+1))-b.x2f(k,j,(is-i+2));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(is-i)) = b.x1f(k,j,(is-i+1))
        +(pco->dx1f(is-i+1)/pco->dx2f(j))
        *(b.x2f(k,(j+1),(is-i+1)) - b.x2f(k,j,(is-i+1)));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(is-i)) = b.x3f(k,j,(is+i-1));
      }
    }}
  }

  return;
}

//==============================================================================
// Open boundary condition at the right edge (outer x1 boundary)
//==============================================================================
void OpenOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
        for (int i=1; i<=ngh; ++i) {
          prim(n,k,j,ie+i) = prim(n,k,j,ie-i+1);
        }
      }
    }
  }

  // inflow restriction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        if (prim(IVX,k,j,ie+i) < 0.0) {
          prim(IVX,k,j,ie+i) = 0.0;
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je+1; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(ie+i)) = 2.0*b.x2f(k,j,(ie+i-1))-b.x2f(k,j,(ie+i-2));
      }
    }}

    for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(ie+i+1)) = b.x1f(k,j,(ie+i))
        -(pco->dx1f(ie+i)/pco->dx2f(j))
        *(b.x2f(k,(j+1),(ie+i)) - b.x2f(k,j,(ie+i)));
      }
    }}

    for (int k=ks; k<=ke+1; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(ie+i)) = b.x3f(k,j,(ie-i+1));
      }
    }}
  }

  return;
}

//==============================================================================
// OpenInnerX2 boundary condition
//==============================================================================
void OpenInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,k,js-j,i) = prim(n,k,js+j-1,i);
        }
      }
    }
  }

  // Inflow restriction
  Real dn_ratio;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,js-j,i) > 0.0) {
          prim(IVY,k,js-j,i) = 0.0;
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,(js-j),i) = 2.0*b.x1f(k,(js-j+1),i) - b.x1f(k,(js-j+2),i);
        }
      }
    }

    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x2f(k,(js-j),i) = b.x2f(k,(js-j+1),i)
          +pco->dx2f(js-j+1)/pco->dx1f(i)*(b.x1f(k,(js-j+1),i+1)-b.x1f(k,(js-j+1),i));
        }
      }
    }

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,(js-j),i) = b.x3f(k,(js+j-1),i);
        }
      }
    }
  }

  return;
}

//==============================================================================
// OpenOuterX2 boundary condition
//==============================================================================
void OpenOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                FaceField &b, Real time, Real dt,
                int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          prim(n,k,je+j,i) = prim(n,k,je-j+1,i);
        }
      }
    }
  }

  // Inflow restriction
  Real dn_ratio;
  for (int k=ks; k<=ke; ++k) {
    for (int j=1; j<=ngh; ++j) {
      for (int i=is; i<=ie; ++i) {
        if (prim(IVY,k,je+j,i) < 0.0) {
          prim(IVY,k,je+j,i) = 0.0;
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          b.x1f(k,(je+j),i) = 2.0*b.x1f(k,(je+j-1),i) - b.x1f(k,(je+j-2),i);
        }
      }
    }

    for (int k=ks; k<=ke; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x2f(k,(je+j+1),i) = b.x2f(k,(je+j),i)
          -pco->dx2f(je+j)/pco->dx1f(i)*(b.x1f(k,(je+j),i+1)-b.x1f(k,(je+j),i));
        }
      }
    }

    for (int k=ks; k<=ke+1; ++k) {
      for (int j=1; j<=ngh; ++j) {
        for (int i=is; i<=ie; ++i) {
          b.x3f(k,(je+j  ),i) = b.x3f(k,(je-j+1),i);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
//  \brief Reduce boundary conditions, inner x3 boundary

void ReduceInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ks-k,j,i) = prim(n,ks,j,i);
        }
      }
    }
  }

  // Reduce v3
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        prim(IVZ,ks-k,j,i) = prim(IVZ,ks-k,j,i);
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f((ks-k),j,i) = b.x1f(ks,j,i);
        }
      }}

    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f((ks-k),j,i) = b.x2f(ks,j,i);
        }
      }}

    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f((ks-k),j,i) = b.x3f(ks,j,i);
        }
      }}
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ReduceOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
//                          FaceField &b, Real time, Real dt,
//                          int is, int ie, int js, int je, int ks, int ke, int ngh)
//  \brief Reduce boundary conditions, outer x3 boundary

void ReduceOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                    FaceField &b, Real time, Real dt,
                    int is, int ie, int js, int je, int ks, int ke, int ngh) {
  // copy hydro variables into ghost zones
  for (int n=0; n<(NHYDRO); ++n) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          prim(n,ke+k,j,i) = prim(n,ke,j,i);
        }
      }
    }
  }

  // Reduce v3 (or pressure)
  for (int k=1; k<=ngh; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        //prim(IVZ,ke+k,j,i) = 0.05*fabs(prim(IVZ,ke,j,i))+prim(IVZ,ke+k-1,j,i);
        if (fabs(b.x3f((ke+1),j,i)) >= 0.1) {
          prim(IPR,ke+k,j,i) = prim(IPR,ke,j,i)*(1.0-k*0.01);
        }
      }
    }
  }

  // copy face-centered magnetic fields into ghost zones
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          b.x1f((ke+k  ),j,i) = b.x1f((ke  ),j,i);
        }
      }}

    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x2f((ke+k  ),j,i) = b.x2f((ke  ),j,i);
        }
      }}

    for (int k=1; k<=ngh; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          b.x3f((ke+k+1),j,i) = b.x3f((ke+1),j,i);
        }
      }}
  }

  return;
}