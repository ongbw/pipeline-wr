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
  pipeline_DNWR.c solves the Heat equation with variable diffusive coefficient
  using a pipeline DNWR algorithm% for 2 to N subdomains
  
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
  double theta; // DNWR parameter
    
  if (argc != 7) {
    if (proc_id == 0) {
      printf("[error] usage: ./executable <Nx> <Nt> <DimX> <# iterates>  <# transmissions> <theta>\n");
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
    theta = atof(argv[6]);
  }
    
  if (proc_id == 0) {
        
    if (DimX < 3) {
      printf("Error: this code only supports DimX > 2\n ");
      fflush(stdout);
      exit(4);
    }
        
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
  int MidDom = DimX/2; //handles both odd and even case using div operator
  int MyK = (proc_id / DimX);
  int Mt = Nt/J; // number of time steps to collect before transmitting
    
  // Boundary conditions
  double gl[2*Mt];  // stores gl and if needed,  du/dx_l
  double gr[2*Mt];  // stores gr and if needed,  du/dx_l.  watch signs!
    
  double sendl[2*Mt];  // variables needed for MPI Communication
  double sendr[2*Mt];  // variables needed for MPI Communication
  double recvl[2*Mt];  // variables needed for MPI Communication
  double recvr[2*Mt];  // variables needed for MPI Communication
    
  // [todo]: might need to update for different isend/irecv
  MPI_Request request1, request2, request3, request4; // variables needed for MPI Send Recv
  MPI_Status status; // variables needed for MPI Send Recv
    
  // writing to files
  char filename[32];
    
  for (int nt=0; nt<2*Mt; nt++) {
    gl[nt] = 0.0;
    gr[nt] = 0.0;
    sendl[nt] = 0.0;
    sendr[nt] = 0.0;
    recvl[nt] = 0.0;
    recvr[nt] = 0.0;
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
    
  // Check whether this processor is doing DD, ND, or DN
  if (MyDom == MidDom) {
    // DD in middle domain
    L[0] = 1.0;
    L[DDNx*DDNx - 1] = 1.0;
  } else {
    if ( MyDom < MidDom) {
      // DN
      L[0] = 1.0; // Dirichlet at left boundary
      L[DDNx*DDNx - 1] = 1.0+2*alpha; // Neumann at right boundary
      L[DDNx*DDNx - 2] = -2*alpha;
    } else {
      // ND
      L[0] = 1.0 + 2*alpha; // Neumann at left boundary
      L[1] = -2*alpha;
      L[DDNx*DDNx - 1] = 1.0; // Dirichlet at right boundary
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
  for (int i=0; i<DDNx; i++){
    u[i] = pow((temp + i ) * dx - 0.5,2) - 0.25;  // u(x,0)= (x-0.5)^2
  }
    
    
  int from; // temporary variable for computer where  message is coming from
  int to; // temporary variable for computing where message ig going
  for (int tj = 0; tj<J; tj++) {
        
    /* RECEIVE as appropriate */
    if (MyDom == MidDom) {
      if (MyK > 0) {
	from = proc_id - DimX;
	//receive updated Dirichlet conditions from left
	MPI_Irecv(recvl,2*Mt,MPI_DOUBLE,from-1,0,MPI_COMM_WORLD,&request1);
	//receive updated Dirichlet conditions from right
	MPI_Irecv(recvr,2*Mt,MPI_DOUBLE,from+1,0,MPI_COMM_WORLD,&request2);
	MPI_Wait(&request1, &status);
	MPI_Wait(&request2, &status);
      }
    }
        
    if (MyDom < MidDom) {
      // recv from right (Neumann updates)
      from = proc_id + 1;
      MPI_Irecv(recvr,2*Mt,MPI_DOUBLE,from,0,MPI_COMM_WORLD,&request2);
      if ( (MyK > 0) && (MyDom >0) ) {
	// receive updated Dirichlet from the left
	from = proc_id - DimX - 1;
	MPI_Irecv(recvl,2*Mt,MPI_DOUBLE,from,0,MPI_COMM_WORLD,&request1);
	MPI_Wait(&request1, &status);
      }
      MPI_Wait(&request2, &status);
    }
        
    if (MyDom > MidDom) {
      // recv from left (Neumann Updates)
      from = proc_id - 1;
      MPI_Irecv(recvl,2*Mt,MPI_DOUBLE,from,0,MPI_COMM_WORLD,&request1);
      if ((MyK > 0) && (MyDom < DimX - 1)) {
	// receive updated Dirichlet Conditions from the right
	from = proc_id - DimX + 1;
	MPI_Irecv(recvr,2*Mt,MPI_DOUBLE,from,0,MPI_COMM_WORLD,&request2);
	MPI_Wait(&request2, &status);
      }
      MPI_Wait(&request1, &status);
    }
    /* end: RECEIVE as appropriate */
        
    /* make sure send buffer is safe to overwrite */
    if (tj > 0) {
      if (MyDom == MidDom) {
	// middle domain sent Neumann updates to left and right
	MPI_Wait(&request3, &status);
	MPI_Wait(&request4, &status);
      }
            
      if (MyDom < MidDom) {
	if (MyK < (K-1)) {
	  // this domain sent Dirichlet updates to right
	  MPI_Wait(&request4, &status);
	}
	if (MyDom > 0) {
	  // this domain sent Neumann updates to left
	  MPI_Wait(&request3, &status);
	}
      }
            
      if (MyDom > MidDom) {
	if (MyK < (K-1)) {
	  // this domain sent Dirichlet update to left
	  MPI_Wait(&request3, &status);
	}
	if (MyDom < (DimX - 1)) {
	  // this domain sent Neumann update to right
	  MPI_Wait(&request4, &status);
	}
                
      }
    }
    /* end: make sure send buffer is safe to overwrite */
        
    /* Overwrite information as necessary */
    if (MyDom == MidDom) {
      if (MyK > 0) {
	// only need the Dirichlet updates even though more sent.
	for (int nt =0; nt<Mt; nt++) {
	  gl[nt] = recvl[nt];
	  gr[nt] = recvr[nt];
	}
      }
    }
        
    if (MyDom < MidDom) {
      // update Neumann (and store Dirichlet)
      for (int nt =0; nt<2*Mt; nt++) {
	gr[nt] = recvr[nt];
      }
      if (MyK > 0) {
	// update Dirichlet
	for (int nt =0; nt<Mt; nt++) {
	  gl[nt] = recvl[nt];
	}
      }
    }
        
    if (MyDom > MidDom) {
      // update Dirichlet and Neumann
      for (int nt =0; nt<2*Mt; nt++) {
	gl[nt] = recvl[nt];
      }
      if (MyK > 0) {
	// update Dirichlet
	for (int nt =0; nt<Mt; nt++) {
	  gr[nt] = recvr[nt];
	}
      }
    }
    /* end: Overwrite information as necessary */
        
    /* Time loop to compute before transmitting */
    for (int nt = 0; nt<Mt; nt++) {
            
      if (MyDom == MidDom) {
	// initialize implicit neumann solve
	sendl[Mt+nt] = u[0];
	sendr[Mt+nt] = u[DDNx-1];
                
	// enforce boundary conditions
	u[0] = gl[nt];
	u[DDNx-1] = gr[nt];
                
	// forward/backwards substitution
	dgetrs(&transpose_flag,&DDNx,&nrhs,L,&DDNx,pivot,u,&DDNx,&info);
                
	// compute Neumann Trace (using ghost point update)
	sendl[Mt+nt] = sendl[Mt+nt] - (2*alpha + 1)*u[0] + 2*alpha*u[1];
	sendr[Mt+nt] = sendr[Mt+nt] - (2*alpha + 1)*u[DDNx - 1] + 2*alpha * u[DDNx-2];
	sendl[nt] = gl[nt];
	sendr[nt] = gr[nt];
      }
            
      if (MyDom < MidDom) {
	// initialize implicit neumann solve
	sendl[Mt+nt] = u[0];
                
	// enforce boundary condition
	u[0] = gl[nt];
	u[DDNx-1] = u[DDNx-1] + gr[Mt+nt];
                
	// forward/backwards substitution
	dgetrs(&transpose_flag,&DDNx,&nrhs,L,&DDNx,pivot,u,&DDNx,&info);
                
	// compute Neumann Trace (using ghost point update)
	sendl[Mt+nt] = sendl[Mt+nt] - (2*alpha + 1)*u[0] + 2*alpha*u[1];
	sendl[nt] = gl[nt];
	sendr[nt] = (1-theta)*gr[nt] + theta*u[DDNx-1];
      }
            
      if (MyDom > MidDom) {
	// initialize implicit neumann solve
	sendr[Mt+nt] = u[DDNx-1];
                
	// enforce boundary condition
	u[0] = u[0] + gl[Mt+nt];
	u[DDNx-1] = gr[nt];
                
	// forward/backwards substitution
	dgetrs(&transpose_flag,&DDNx,&nrhs,L,&DDNx,pivot,u,&DDNx,&info);
                
	// compute Neumann Trace (using ghost point update)
	sendr[Mt+nt] = sendr[Mt+nt] - (2*alpha + 1)*u[DDNx - 1] + 2*alpha * u[DDNx-2];
	sendl[nt] = (1-theta)*gl[nt] + theta*u[0];
	sendr[nt] = gr[nt];
      }      
    }  
    /* End: Time loop to compute before transmitting */
        
    /* send appropriate traces with neighbouring domains */
    if (MyDom == MidDom) {
      to = proc_id-1;
      MPI_Isend(sendl,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request3);
      to = proc_id+1;
      MPI_Isend(sendr,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request4);
    }
        
    if (MyDom < MidDom) {
      if (MyK < (K-1)) {
	to = proc_id + 1 + DimX;
	MPI_Isend(sendr,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request4);
      }
      if (MyDom > 0) {
	to = proc_id-1;
	MPI_Isend(sendl,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request3);
      }
    }
        
    if (MyDom > MidDom) {
      if (MyK < (K-1)) {
	to = proc_id - 1 + DimX;
	MPI_Isend(sendl,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request3);
      }
      if (MyDom < (DimX - 1)) {
	to = proc_id+1;
	MPI_Isend(sendr,2*Mt,MPI_DOUBLE,to,0,MPI_COMM_WORLD,&request4);
      }      
    }
    /* end:send appropriate traces with neighbouring domains */
        
  } // END: Main Loop
    
  /*
  // output data to file.
  sprintf(filename,"N_DimX_K.%d.%d.%d.dat",DimX,MyDom,MyK);
  FILE *fout;
  fout = fopen(filename,"w");
  temp = MyDom * (DDNx - 1);
  for (int i =0; i<DDNx; i++) {
    fprintf(fout,"%g,%g\n",(temp +i)*dx,u[i]);
  }
  fclose(fout);
  */
  
  // free variables
  free(L);
  free(pivot);

  
  MPI_Barrier(MPI_COMM_WORLD);

  /* Collect the data into the variables passed in */
  if((retval=PAPI_flops( &real_time, &proc_time, &flpins, &mflops))<PAPI_OK)
    test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
  
  printf("Real_time: %f\nProc_time: %f\nTotal flpins: %lld\nMFLOPS: %f\n",
	 real_time, proc_time, flpins, mflops);
  printf("%s\tPASSED\n", __FILE__);
  PAPI_shutdown();


  /*
  if (proc_id ==0) {
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("[proc %d]: %d iterations took %d microseconds\n",proc_id,K,msec);
  }      

  if (proc_id ==(nprocs-DimX)) {
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("[proc %d]: %d iterations took %d microseconds\n",proc_id,K,msec);
  }

  if (proc_id ==(nprocs-1)) {
    diff = clock() - start;
    int msec = diff * 1000 / CLOCKS_PER_SEC;
    printf("[proc %d]: %d iterations took %d microseconds\n",proc_id,K,msec);
  }
  */
  
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

