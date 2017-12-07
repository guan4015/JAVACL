package MonteCarlo;

import static org.jocl.CL.CL_CONTEXT_PLATFORM;
import static org.jocl.CL.CL_DEVICE_TYPE_GPU;
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
import java.util.ArrayList;

public class CL_PriceGeneration {
	
	private float _interest;
	private float _volatility;
	private float[] _gaussianVec;
	private float _initialValue;
	private float _duration;
	/**
	 * Constructors
	 * @param option
	 * @param gaussianVec
	 */
	public CL_PriceGeneration( Option<?,?> option, float[] gaussianVec ) {
		_interest = (float) option.getInterestRate();
		_volatility = (float) option.getVolatility();
		_initialValue = (float) option.getStartPrice();
		_duration = (float) option.getDuration();
		_gaussianVec = gaussianVec;	
	}
	
	
	/**
	 * The following function generates stock prices
	 * @param normalVec normal random vectors
	 * @param r interest rate
	 * @param v volatility
	 * @return
	 */
	public float[] StockPrices() {
		
		
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
        // Initialize the values
		float r = _interest;
		float v = _volatility;
		float S0 = _initialValue;
		float T = _duration;
        
		String src = "__kernel void geoBrownian(__global float* out)\n"
				+ "{\n" 
				+ "    int i = get_global_id(0);\n" 
				+ "    out[i] = " + S0 + "*exp((" + r + " - " + v + " * " + v + "/2)* " + T 
				+ "    +" + v + "*out[i] * sqrt(" + T +")); \n" 
				+ "}";
		

        // Create the program from the source code
        cl_program program = clCreateProgramWithSource( context,
                1, new String[]{ src }, null, null );

        // Build the program
        clBuildProgram( program, 0, null, null, null, null );

        // Create the kernel
        cl_kernel kernel = clCreateKernel( program, "geoBrownian", null );

        
        // Set the length
        int length = _gaussianVec.length;

        // Set the pointers
        Pointer norm = Pointer.to( _gaussianVec );

        // Allocate the memory objects for the input- and output data
        
        cl_mem memObject = clCreateBuffer( context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                Sizeof.cl_float * length, norm, null );

        // System.out.println( System.currentTimeMillis() - tmp );

        // Set the arguments for the kernel
        clSetKernelArg( kernel, 0,
                Sizeof.cl_mem, Pointer.to( memObject ) );

        // System.out.println((System.currentTimeMillis() - tmp));

        // Set the work-item dimensions
        long global_work_size[] = new long[]{ length };
        long local_work_size[] = new long[]{1};

        // Execute the kernel
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                global_work_size, local_work_size, 0, null, null);

        // Read the output data
        clEnqueueReadBuffer( commandQueue, memObject, CL_TRUE, 0,
        		length * Sizeof.cl_float, norm, 0, null, null );
     
        // pull out the results
        return _gaussianVec;

	}
	
	public static void main(String[] args) {
		// Test this method
		Option<Integer, Float> IBM_eu = new Option<Integer, Float>("IBM","European",0.0001,152.35,0.01,165);
		IBM_eu.setDuration( (float) 252 );
		CL_Generation normVec = new CL_Generation( 10000 );
		ArrayList<float[]> output = normVec.getGaussianVector();
		CL_PriceGeneration price = new CL_PriceGeneration( IBM_eu, output.get(0) );
		float[] price_out = price.StockPrices();
		// The results could be printed on screen
		for (int i = 0; i < price_out.length; i++ ) {
			System.out.println( price_out[i] );
		}
		
	}

}
