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
pipeline_NNWR.c solves the Heat equation with variable diffusive coefficient 
using a pipeline NNWR algorithm% for 2 to N subdomains 

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
  int J; // number of transmissions in [0,T]
  
  if (argc != 6) {
    if (proc_id == 0) {
      printf("[error] usage: ./executable <Nx> <Nt> <DimX> <# iterates>  <# transmissions><\n");
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
    J = atoi(argv[5]);
  }

  if (proc_id == 0) {
    // assume for now, that some integer multiple of 2*DimX processors are requested

    if ( nprocs != (K*DimX)  ) {
      printf("Error: please request K*DimX processors for this job\n ");
      fflush(stdout);
      exit(1);
    }
    
    // check if Nx divides into subdomains equally
    if ( Nx % DimX != 0) {
      printf("Error: Nx is not divisible by DimX.  Exiting... \n");
      fflush(stdout);
      exit(2);
    }
    
    // check if time domain can be split equally into J parts
    if ( Nt % J != 0) {
      printf("Error: Nt is not divisible by J.  Exiting... \n");
      exit(3);
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
  int DDNx = Nx/DimX + 1; // number of nodes in each subdomain in x dir

  // Initial condition
  double u[DDNx];
  int MyDom = (proc_id % DimX);
  int MyK = (proc_id / DimX);
  int Mt = Nt/J; // number of time steps to collect before transmitting
    
  // Boundary conditions
  double g[4*Mt];  // stores gl, gr, dphi/dx_l, dphi/dx_r
 
  double recvg[4*Mt]; // variables needed for MPI Send Recv
  MPI_Request request, request2; // variables needed for MPI Send Recv
  MPI_Status status; // variables needed for MPI Send Recv

  // writing to files
  char filename[32];
  
  // second data structure that's needed to keep "old" dirichlet trace
  double tracel[Mt];
  double tracer[Mt];
  double recv_tracel[Mt];
  double recv_tracer[Mt];
  
  for (int nt=0; nt<4*Mt; nt++) {
    g[nt] = 0.0;
  }

  for (int nt = 0; nt<Mt; nt++) {
    tracel[nt] = 0.0;
    tracer[nt] = 0.0;
  }
    
  // define matrix, L = (1 - dt*Laplace), for linear solve at each time step
  double * L ;
  L = (double *) calloc(DDNx*DDNx , sizeof(double));

  int ind;
  double alpha;
  alpha = dt/dx/dx;
  double temp1, temp2;
  temp1 = 1.0 + 2*alpha;
  temp2 = -alpha;

  for (int i = 1; i<(DDNx-1); i++) {
    ind = i*DDNx + i;
    L[ind] = temp1;
    L[ind-1] = temp2;
    L[ind+1] = temp2;
  }

  // Check whether this processor is doing DD or "NN"
  if (MyK % 2 == 0) {
    // This is Dirichlet
    L[0] = 1.0;
    L[DDNx*DDNx - 1] = 1.0;
  } else {
    if ( MyDom == 0) {
      L[0] = 1.0; // Dirichlet at domain boundary
    } else {
      // one sided differencing
      //L[0] = 1.0;
      //L[1] = -1.0;
      // ghost points
      L[0] = 1.0 + 2*alpha;
      L[1] = -2*alpha;
    }
    if ( MyDom == (DimX-1) ) {
      L[DDNx*DDNx - 1] = 1.0; // Dirichlet at domain boundary
    } else {
      // one sided differencing
      //L[DDNx*DDNx - 1] = 1.0;
      //L[DDNx*DDNx - 2] = -1.0;
      
      // ghost points
      L[DDNx*DDNx - 1] = 1.0+2*alpha;
      L[DDNx*DDNx - 2] = -2*alpha;
    }
  }
  
  int info;
  int *pivot=(int *) calloc(DDNx, sizeof(int));

  // Factorize operator L.
  dgetrf(&DDNx,&DDNx,L,&DDNx,pivot,&info);

  char transpose_flag = 'T';
  int nrhs = 1;

  int temp = MyDom * (DDNx-1);

  // specify initial condition
  if (MyK % 2 == 0 ) {
    for (int i=0; i<DDNx; i++){
      u[i] = pow((temp + i ) * dx - 0.5,2) - 0.25;  // u(x,0)= (x-0.5)^2
    }
  } else {
    for (int i=0; i<DDNx; i++){
      u[i] = 0.0;
    }
  }

  for (int tj = 0; tj<J; tj++) {
    
    if (MyK > 0) {
      // receive appropriate boundary condition
      int from = (proc_id - DimX) % nprocs;
      if (from < 0) {
	from = from + nprocs;
      }
      
      MPI_Irecv(recvg,4*Mt,MPI_DOUBLE,from,0,MPI_COMM_WORLD,&request2);
      MPI_Wait(&request2, &status);
      
      if ( (tj > 0) && (MyK < (K-1) ) ){
	MPI_Wait(&request, &status);
      }
      
      for (int nt =0; nt < 4*Mt; nt++) {
	g[nt] = recvg[nt];
      }
    }
    
      
    if (MyK % 2 == 0 ) {
      // DD step
      for (int nt = 0; nt<Mt; nt++) {

	if (MyDom > 0) {
	  tracel[nt] = u[0];
	}
	if (MyDom < DimX - 1) {
	  tracer[nt] = u[DDNx - 1];
	}
	
	// enforce boundary conditions before linear solve
	u[0] = g[nt];
	u[DDNx-1] = g[2*Mt + nt];
	
	// Forward/backwards substitution (linear solve)
	dgetrs(&transpose_flag,&DDNx,&nrhs,L,&DDNx,pivot,u,&DDNx,&info);
	
	// compute Neumann Trace (using ghost point update)
	if (MyDom > 0) {
	  tracel[nt] = -tracel[nt] + (2*alpha + 1)*u[0] - 2*alpha*u[1];
	} 
	if (MyDom < DimX - 1) {
	  tracer[nt] = -tracer[nt] + (2*alpha + 1)*u[DDNx - 1] - 2*alpha * u[DDNx-2];
	}
      }
    } else {

      // NN step
      for (int nt = 0; nt<Mt; nt++) {
	
	// enforce boundary conditions
	if (MyDom > 0) {
	   u[0] = u[0] + g[Mt+nt];
	}
	
	if (MyDom < DimX - 1) {
	  u[DDNx-1] = u[DDNx-1] + g[3*Mt+nt];
	}
	
	// forward/backwards substitution
	dgetrs(&transpose_flag,&DDNx,&nrhs,L,&DDNx,pivot,u,&DDNx,&info);
	
	// assemble Dirichlet Trace
	tracel[nt] = u[0];
	tracer[nt] = u[DDNx-1];
      } 
    }


    
    // send recv appropriate traces with neighbouring domains
    if (MyDom > 0) {
      MPI_Sendrecv(tracel,Mt,MPI_DOUBLE,proc_id - 1,MyK,
		   recv_tracel,Mt,MPI_DOUBLE,proc_id -1,MyK,
		   MPI_COMM_WORLD, &status);
    }
    
    if (MyDom < DimX - 1) {
      MPI_Sendrecv(tracer,Mt,MPI_DOUBLE,proc_id + 1,MyK,
		   recv_tracer,Mt,MPI_DOUBLE,proc_id +1,MyK,
		   MPI_COMM_WORLD, &status) ;
    } //end send recv to neighbouring domains
    
      // compute updated bc (to transmit)
    if (MyK % 2 == 0) {
      if (MyDom > 0) {
	for (int nt = 0; nt<Mt; nt++) {
	  g[Mt + nt] = tracel[nt] + recv_tracel[nt];
	}
      } else {
	for (int nt = 0; nt<Mt; nt++) {
	  g[Mt+nt] = 0.0;
	}
      }
      if (MyDom < DimX - 1) {
	for (int nt = 0; nt<Mt; nt++) {
	  g[3*Mt + nt] = tracer[nt] + recv_tracer[nt];
	}
      } else {
	for (int nt = 0; nt<Mt; nt++) {
	  g[3*Mt+nt] = 0.0;
	}
      }
    } else {
      if (MyDom > 0) {
	for (int nt = 0; nt<Mt; nt++) {
	  // theta = 0.25
	  g[nt] = g[nt] - 0.25 * (tracel[nt] + recv_tracel[nt]);
	}
      }      
      if (MyDom < DimX - 1) {
	for (int nt = 0; nt<Mt; nt++) {
	  g[2*Mt + nt] = g[2*Mt+ nt] - 0.25 * (tracer[nt] + recv_tracer[nt]);
	}
      }
    }
    
    if (MyK < (K-1)) {
      // send appropriate boundary conditions
      // specifically, send to proc_id + DimX
      int to = (proc_id + DimX) % nprocs;
      MPI_Isend(g,4*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request);
    }
    
  }


  /*
  if (MyK % 2 == 0) {
    // output data to file.
    sprintf(filename,"N_DimX_K.%d.%d.%d.dat",DimX,MyDom,MyK);
    FILE *fout;
    fout = fopen(filename,"w");
    int temp = MyDom * (DDNx - 1);
    for (int i =0; i<DDNx; i++) {
      fprintf(fout,"%g,%g\n",(temp +i)*dx,u[i]);
    }
    fclose(fout);
    }
*/
  
  
  // free variables
  free(L);
  free(pivot);

  if (proc_id ==0) {
    double diff;
    time(&stop);
    diff = difftime(stop,start);
    printf("[proc %d]: %d iterations took %g seconds\n",proc_id,K,diff);
  }      

  if (proc_id ==(nprocs-1)) {
    double diff;
    time(&stop);
    diff = difftime(stop,start);
    printf("[proc %d]: %d iterations took %g seconds\n",proc_id,K,diff);
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

