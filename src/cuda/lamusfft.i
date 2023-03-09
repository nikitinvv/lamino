/*interface*/
%module lamusfft

%{
#define SWIG_FILE_WITH_INIT
#include "lamusfft.cuh"
%}

class lamusfft {

public:
  %immutable;
  size_t n0,n1,n2;  
  size_t detw,deth;  
  size_t ntheta; // number of angles
  float phi;

  %mutable;  
  lamusfft(size_t n0, size_t n1, size_t n2, size_t detw, size_t deth, size_t ntheta, float phi, float eps);
  ~lamusfft();
  void fwd(size_t g, size_t f, size_t theta);
  void adj(size_t f, size_t g, size_t theta);
  void free();
};