package MonteCarlo;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_MAX_COMPUTE_UNITS;
import static org.jocl.CL.CL_DEVICE_NAME;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
import static org.jocl.CL.CL_DEVICE_VENDOR;
import static org.jocl.CL.CL_DRIVER_VERSION;
import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_READ_WRITE;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateCommandQueue;
import static org.jocl.CL.clCreateContext;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clSetKernelArg;

import java.util.ArrayList;


import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;

public class CL_Generation {
	
	private int _trial;
	ArrayList<float[]> _gaussianVec; 
	
	public CL_Generation( int trial ) {
		_trial = trial;
		_gaussianVec = new ArrayList<float[]>();
	}
	
	/**
	 * This method generates the stock paths using Java CL
	 * @param trial
	 * @return
	 */
	public ArrayList<float[]> getGaussianVector() {
		
		int num = _trial;
		
		// In what follows, we generate the paths
		cl_platform_id[] platforms = new cl_platform_id[1];      
        clGetPlatformIDs( 1, platforms, null );      
        cl_platform_id platform = platforms[0];
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs( platform,  CL_DEVICE_TYPE_GPU, 1, devices, null );       
        cl_device_id device = devices[0];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty( CL_CONTEXT_PLATFORM, platform );

        // Create a context for the selected device
        cl_context context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null );

        // Create a command-queue for the selected device
        @SuppressWarnings( "deprecation" )
		cl_command_queue commandQueue =
                clCreateCommandQueue( context, device, 0, null );

        // Read the program sources and compile them :
		// Read the program sources and compile them :
		String src = "__kernel void gaussian(__global const float* a, __global const float* b, __global float* out1, __global float* out2)\n " +
				"{\n" + 
				"   int i = get_global_id(0);\n" +
				"   out1[i] = sqrt(-2*log(a[i]))*cos(2*b[i]*3.1415926);\n" +
				"   out2[i] = sqrt(-2*log(a[i]))*sin(2*b[i]*3.1415926);\n" +
			    "}\n";
		

        // Create the program from the source code
        cl_program program = clCreateProgramWithSource( context,
                1, new String[]{ src }, null, null );

        // Build the program
        clBuildProgram( program, 0, null, null, null, null );

        // Create the kernel
        cl_kernel kernel = clCreateKernel( program, "gaussian", null );

        
        // Define the vectors
        float uniformA[] = new float[num];
        float uniformB[] = new float[num];
        float gaussianA[] = new float[num];
        float gaussianB[] = new float[num];
        
        // Generate two sequences of uniform random variables
        for ( int i = 0; i < num; i++ )
        {
            uniformA[i] = (float) Math.random();
            uniformB[i] = (float) Math.random();
        }
        
        // Set the pointers
        Pointer uniA = Pointer.to( uniformA );
        Pointer uniB = Pointer.to( uniformB );
        Pointer gauA = Pointer.to( gaussianA );
        Pointer gauB = Pointer.to( gaussianB );

        // Allocate the memory objects for the input- and output data
        cl_mem memObjects[] = new cl_mem[5];
        
        memObjects[0] = clCreateBuffer( context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * num, uniA, null );
        memObjects[1] = clCreateBuffer( context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * num, uniB, null );
        memObjects[2] = clCreateBuffer( context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * num, null, null );
        memObjects[3] = clCreateBuffer( context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * num, null, null );

        // System.out.println( System.currentTimeMillis() - tmp );

        // Set the arguments for the kernel
        clSetKernelArg( kernel, 0,
                Sizeof.cl_mem, Pointer.to( memObjects[0] ) );
        clSetKernelArg( kernel, 1,
                Sizeof.cl_mem, Pointer.to( memObjects[1] ) );
        clSetKernelArg( kernel, 2,
                Sizeof.cl_mem, Pointer.to( memObjects[2] ) );
        clSetKernelArg( kernel, 3,
                Sizeof.cl_mem, Pointer.to( memObjects[3] ) );


        // System.out.println((System.currentTimeMillis() - tmp));

        // Set the work-item dimensions
        long global_work_size[] = new long[]{ num };
        long local_work_size[] = new long[]{1};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer( commandQueue, memObjects[3], CL_TRUE, 0,
                num * Sizeof.cl_float, gauB, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[2], CL_TRUE, 0,
                num * Sizeof.cl_float, gauA, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[0], CL_TRUE, 0,
                num * Sizeof.cl_float, uniA, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[1], CL_TRUE, 0,
                num * Sizeof.cl_float, uniB, 0, null, null );
        
        // pull out the results
        ArrayList<float[]> results = new ArrayList<float[]>();
        results.add( gaussianA );
        results.add( gaussianB );
        _gaussianVec = results;
        return results;

	}
	
	/**
	 * 
	 * @param normalVec normal random vectors
	 * @param r interest rate
	 * @param v volatility
	 * @return
	 */
	public float[] StockBrownianPath( float[] normalVec, float[] r, float[] v ) {
		
		
		// In what follows, we generate the paths
		cl_platform_id[] platforms = new cl_platform_id[1];      
        clGetPlatformIDs( 1, platforms, null );      
        cl_platform_id platform = platforms[0];
        cl_device_id[] devices = new cl_device_id[1];
        clGetDeviceIDs( platform,  CL_DEVICE_TYPE_GPU, 1, devices, null );       
        cl_device_id device = devices[0];

        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty( CL_CONTEXT_PLATFORM, platform );

        // Create a context for the selected device
        cl_context context = clCreateContext(
                contextProperties, 1, new cl_device_id[]{device},
                null, null, null );

        // Create a command-queue for the selected device
        @SuppressWarnings( "deprecation" )
		cl_command_queue commandQueue =
                clCreateCommandQueue( context, device, 0, null );

        // Read the program sources and compile them :
		// Read the program sources and compile them :
		String src2 = "__kernel void geoBrownian(__global float* r, __global float* v, __global float* c, __global float* out3)\n"
				+ "{\n" 
				+ "    int i = get_global_id(0);\n" 
				+ "    if (i >= n-1)\n" 
				+ "        return;\n" 
				+ "\n"
				+ "out3[0] = exp(r[0] - v[0]*v[0]/2 + v[0]*c[0]);\n"
				+ "out3[i+1] = out3[i]*exp(r[0] - v[0]*v[0]/2 + v[0]*c[i]);\n"
				+ "}";
		

        // Create the program from the source code
        cl_program program = clCreateProgramWithSource( context,
                1, new String[]{ src2 }, null, null );

        // Build the program
        clBuildProgram( program, 0, null, null, null, null );

        // Create the kernel
        cl_kernel kernel = clCreateKernel( program, "geoBrownian", null );

        
        // Define the vectors
        int length = normalVec.length + 1;
        float prices[] = new float[ length ];


        // Set the pointers
        Pointer norm = Pointer.to( normalVec );
        Pointer stockPrice = Pointer.to( prices );
        Pointer interest = Pointer.to( r );
        Pointer volatility = Pointer.to( v );
        


        // Allocate the memory objects for the input- and output data
        cl_mem memObjects[] = new cl_mem[5];
        
        memObjects[0] = clCreateBuffer( context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * length, interest, null );
        memObjects[1] = clCreateBuffer( context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * length, volatility, null );
        memObjects[2] = clCreateBuffer( context,
        		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * length, norm, null );
        memObjects[3] = clCreateBuffer( context,
                CL_MEM_READ_WRITE,
                Sizeof.cl_float * length, null, null );

        // System.out.println( System.currentTimeMillis() - tmp );

        // Set the arguments for the kernel
        clSetKernelArg( kernel, 0,
                Sizeof.cl_mem, Pointer.to( memObjects[0] ) );
        clSetKernelArg( kernel, 1,
                Sizeof.cl_mem, Pointer.to( memObjects[1] ) );
        clSetKernelArg( kernel, 2,
                Sizeof.cl_mem, Pointer.to( memObjects[2] ) );
        clSetKernelArg( kernel, 3,
                Sizeof.cl_mem, Pointer.to( memObjects[3] ) );


        // System.out.println((System.currentTimeMillis() - tmp));

        // Set the work-item dimensions
        long global_work_size[] = new long[]{ length };
        long local_work_size[] = new long[]{1};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer( commandQueue, memObjects[3], CL_TRUE, 0,
        		length * Sizeof.cl_float, stockPrice, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[2], CL_TRUE, 0,
        		length * Sizeof.cl_float, norm, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[0], CL_TRUE, 0,
        		length * Sizeof.cl_float, interest, 0, null, null );
        clEnqueueReadBuffer( commandQueue, memObjects[1], CL_TRUE, 0,
        		length * Sizeof.cl_float, volatility, 0, null, null );
        
        // pull out the results
        return prices;

	}
	

}
