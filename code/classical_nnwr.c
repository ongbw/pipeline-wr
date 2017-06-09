#include "mpi.h"
#include "math.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "mkl.h"
#include "mkl_blas.h"
#include "mkl_lapacke.h"
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <memory.h>
#include <malloc.h>
#include "papi.h"

#define INDEX 100

extern void test_fail(char *file, int line, char *call, int retval);



/*
classicalNNWR.c solves the Heat equation with variable diffusive coefficient 
using a classical NNWR algorithm% for 2 to N subdomains 

     u_t - u_xx = 0,
     u(x,0) = u_0(x)
     u(a,t) = 0, u(b,t) = 0.
     u_0(x) = (x-0.5)^2 -0.25
     x in [0, 1]
*/ 


int main (int argc, char* argv[]) {

  time_t start;
  time_t stop;
  time(&start);
  
  int nprocs, proc_id;
  
  MPI_Init (&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  
  int Nx; /* Spatial Discretization */
  int Nt; /* Tempmoral Discretization */  
  int DimX; // number of subdomains
  int K; // number of waveform iterates
  
  if (argc != 5) {
    if (proc_id == 0) {
      printf("[error] usage: ./executable <Nx> <Nt> <DimX> <# iterates> \n");
      printf("... Terminating job ...\n");
      fflush(stdout);
    }
    MPI_Finalize();
    exit(0);
  } else {
    Nx = atoi(argv[1]); // number of intervals
    Nt = atoi(argv[2]);
    DimX = atoi(argv[3]);
    K = atoi(argv[4]);
  }

  if (proc_id == 0) {

    if ( nprocs != (DimX)  ) {
      printf("Error: please request DimX processors for this job\n ");
      fflush(stdout);
      exit(1);
    }
    
    // check if Nx divides into subdomains equally
    if ( Nx % DimX != 0) {
      printf("Error: Nx is not divisible by DimX.  Exiting... \n");
      fflush(stdout);
      exit(2);
    }
    
  }

  
  // using PAPi to measure flops
  extern void dummy(void *);
  float real_time, proc_time, mflops;
  long long flpins;
  int retval;

  /* Setup PAPI library and begin collecting data from the counters */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);


  /* Parameter Specification */
  double a = 0.0; // domain of interest Omega=[a,b]
  double b = 1.0;
  double ti = 0.0;
  double tf = 0.1;  // time interval [ti,tf]
  double dt = (double) (tf-ti)/Nt;
  double dx = (double) (b-a)/Nx;
  int DDNx = Nx/DimX + 1; // number of nodes in each subdomain

  // Initial condition
  double u[DDNx];
  double phi[DDNx];
  int MyDom = (proc_id % DimX);
    
  // Boundary conditions
  double g[4*Nt];  // stores gl, gr, dphi/dx_l, dphi/dx_r
 
  MPI_Request request, request2; // variables needed for MPI Send Recv
  MPI_Status status; // variables needed for MPI Send Recv

  // writing to files
  char filename[32];
  
  // second data structure that's needed to keep "old" dirichlet trace
  double tracel[Nt];
  double tracer[Nt];
  double recv_tracel[Nt];
  double recv_tracer[Nt];
  
  for (int nt=0; nt<4*Nt; nt++) {
    g[nt] = 0.0;
  }

  for (int nt = 0; nt<Nt; nt++) {
    tracel[nt] = 0.0;
    tracer[nt] = 0.0;
  }
    
  // define matrix, D (or N) = (1 - dt*Laplace), for linear solve at each time step
  double * D ;
  D = (double *) calloc(DDNx*DDNx , sizeof(double));

  double * N ;
  N = (double *) calloc(DDNx*DDNx , sizeof(double));

  int ind;
  double alpha;
  alpha = dt/dx/dx;
  double temp1, temp2;
  temp1 = 1.0 + 2*alpha;
  temp2 = -alpha;

  for (int i = 1; i<(DDNx-1); i++) {
    ind = i*DDNx + i;
    D[ind] = temp1;
    D[ind-1] = temp2;
    D[ind+1] = temp2;
    N[ind] = temp1;
    N[ind-1] = temp2;
    N[ind+1] = temp2;
  }


  D[0] = 1.0;
  D[DDNx*DDNx - 1] = 1.0;
  
  if ( MyDom == 0) {
    N[0] = 1.0; // Dirichlet at domain boundary
  } else {
    N[0] = 1.0 + 2*alpha;
    N[1] = -2*alpha;
  }

  if ( MyDom == (DimX-1) ) {
    N[DDNx*DDNx - 1] = 1.0; // Dirichlet at domain boundary
  } else {
    N[DDNx*DDNx - 1] = 1.0+2*alpha;
    N[DDNx*DDNx - 2] = -2*alpha;
  }
  
  
  int info;
  int *pivotD=(int *) calloc(DDNx, sizeof(int));
  int *pivotN=(int *) calloc(DDNx, sizeof(int));

  // Factorize operator D/N.
  dgetrf(&DDNx,&DDNx,D,&DDNx,pivotD,&info);
  dgetrf(&DDNx,&DDNx,N,&DDNx,pivotN,&info);

  char transpose_flag = 'T';
  int nrhs = 1;

  int temp = MyDom * (DDNx-1);

  for (int k =0; k<K; k++) {

      // specify initial condition
    for (int i=0; i<DDNx; i++){
      u[i] = pow((temp + i ) * dx - 0.5,2) - 0.25;  // u(x,0)= (x-0.5)^2
      phi[i] = 0.0;
    }
    

    // Dirichlet step
    for (int nt = 0; nt<Nt; nt++) {
      
      if (MyDom > 0) {
	tracel[nt] = u[0];
      }
      if (MyDom < DimX - 1) {
	tracer[nt] = u[DDNx - 1];
      }
	
      // enforce boundary conditions before linear solve
      u[0] = g[nt];
      u[DDNx-1] = g[2*Nt + nt];
      
      // Forward/backwards substitution (linear solve)
      dgetrs(&transpose_flag,&DDNx,&nrhs,D,&DDNx,pivotD,u,&DDNx,&info);
      
      // compute Neumann Trace (using ghost point update)
      if (MyDom > 0) {
	tracel[nt] = -tracel[nt] + (2*alpha + 1)*u[0] - 2*alpha*u[1];
      } 
      if (MyDom < DimX - 1) {
	tracer[nt] = -tracer[nt] + (2*alpha + 1)*u[DDNx - 1] - 2*alpha * u[DDNx-2];
      }
    }

    // send recv appropriate traces with neighbouring domains
    if (MyDom > 0) {
      MPI_Sendrecv(tracel,Nt,MPI_DOUBLE,proc_id - 1,k,
		   recv_tracel,Nt,MPI_DOUBLE,proc_id -1,k,
		   MPI_COMM_WORLD, &status);
    }
    
    if (MyDom < DimX - 1) {
      MPI_Sendrecv(tracer,Nt,MPI_DOUBLE,proc_id + 1,k,
		   recv_tracer,Nt,MPI_DOUBLE,proc_id +1,k,
		   MPI_COMM_WORLD, &status) ;
    } //end send recv to neighbouring domains


    if (MyDom > 0) {
      for (int nt = 0; nt<Nt; nt++) {
	g[Nt + nt] = tracel[nt] + recv_tracel[nt];
      }
    } else {
      for (int nt = 0; nt<Nt; nt++) {
	g[Nt+nt] = 0.0;
      }
    }
    
    if (MyDom < DimX - 1) {
      for (int nt = 0; nt<Nt; nt++) {
	g[3*Nt + nt] = tracer[nt] + recv_tracer[nt];
      }
    } else {
      for (int nt = 0; nt<Nt; nt++) {
	g[3*Nt+nt] = 0.0;
      }
    }
    
    
    // Neumann step
    for (int nt = 0; nt<Nt; nt++) {
	
      // enforce boundary conditions
      if (MyDom > 0) {
	phi[0] = phi[0] + g[Nt+nt];
      }
	
      if (MyDom < DimX - 1) {
	phi[DDNx-1] = phi[DDNx-1] + g[3*Nt+nt];
      }
      
      // forward/backwards substitution
      dgetrs(&transpose_flag,&DDNx,&nrhs,N,&DDNx,pivotN,phi,&DDNx,&info);
      
      // assemble Dirichlet Trace
      tracel[nt] = phi[0];
      tracer[nt] = phi[DDNx-1];
    }

    // send recv appropriate traces with neighbouring domains
    if (MyDom > 0) {
      MPI_Sendrecv(tracel,Nt,MPI_DOUBLE,proc_id - 1,k,
		   recv_tracel,Nt,MPI_DOUBLE,proc_id -1,k,
		   MPI_COMM_WORLD, &status);
    }
    
    if (MyDom < DimX - 1) {
      MPI_Sendrecv(tracer,Nt,MPI_DOUBLE,proc_id + 1,k,
		   recv_tracer,Nt,MPI_DOUBLE,proc_id +1,k,
		   MPI_COMM_WORLD, &status) ;
    } //end send recv to neighbouring domains

    if (MyDom > 0) {
      for (int nt = 0; nt<Nt; nt++) {
	// theta = 0.25
	g[nt] = g[nt] - 0.25 * (tracel[nt] + recv_tracel[nt]);
      }
    }      

    if (MyDom < DimX - 1) {
      for (int nt = 0; nt<Nt; nt++) {
	g[2*Nt + nt] = g[2*Nt+ nt] - 0.25 * (tracer[nt] + recv_tracer[nt]);
      }
    }

  } // loop over K
  
  
  // free variables
  free(D);
  free(N);
  free(pivotD);
  free(pivotN);

  if (proc_id ==0) {
    double diff;
    time(&stop);
    diff = difftime(stop,start);
    printf("[proc %d]: %d iterations took %g seconds\n",proc_id,K,diff);

  /*
    for (int i = 0; i<DDNx; i++) {
      printf("%g\n",phi[i]);
    }
  */
  }

  MPI_Barrier(MPI_COMM_WORLD);

  /* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
  
  printf("Real_time: %f\nProc_time: %f\nTotal flpins: %lld\nMFLOPS: %f\n",
	 real_time, proc_time, flpins, mflops);
  printf("%s\tPASSED\n", __FILE__);
  PAPI_shutdown();

  

  fflush(stdout);
  
  MPI_Finalize();
  return 0;
}

static void test_fail(char *file, int line, char *call, int retval){
  printf("%s\tFAILED\nLine # %d\n", file, line);
  if ( retval == PAPI_ESYS ) {
    char buf[128];
    memset( buf, '\0', sizeof(buf) );
    sprintf(buf, "System error in %s:", call );
    perror(buf);
  }
  else if ( retval > 0 ) {
    printf("Error calculating: %s\n", call );
  }
  else {
    printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
  }
  printf("\n");
  exit(1);
}

