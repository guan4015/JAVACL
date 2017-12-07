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

public class Simulation_CL {
	
	@SuppressWarnings("rawtypes")
	
	// Define member data
	protected Option _option;
	protected double _startPrice;
	protected double _strikePrice;
	protected double _probability;
	protected double _interestRate;
	protected double _error;
	protected String _type;
	protected float _length = 252;
	protected int _seed;
	protected int _num_trial = 0;
	
	// Default constructor
	public Simulation_CL() {}
	
	// Constructor receives parameters
	public Simulation_CL(Option<?,?> option, float length, double probability, double error) {
		this._startPrice = option.getStartPrice(); // initial underlying price
		this._strikePrice = option.getStrikePrice();  // strike price of the option
		this._probability = probability;  // p-value (threshold)
		this._error = error;  // two sided statistics error
		this._length = length;  // time period
		this._type = option.getPayOutType();  // payout type "European" or "Asian"
		this._option = option;
		this._interestRate = option.getInterestRate();  // Interest rate

	}
	

	
	public ArrayList<float[]> stockPriceGeneration( int samples ){
		// In what follows, we first generate the normal random vectors
		// Define a normal random variable generation that can generate 1000000 samples
		CL_Generation normvec = new CL_Generation( samples );
		ArrayList<float[]> normalOutput = normvec.getGaussianVector();
		// Output the prices
		ArrayList<float[]> priceOutput = new ArrayList<float[]>();
		for ( int i = 0; i < normalOutput.size(); i++ ) {
			CL_PriceGeneration output = new CL_PriceGeneration( _option, normalOutput.get(i) );
			priceOutput.add(output.StockPrices());
		}
		return priceOutput;
	}
	


	public double simulate() {
		// compute the stopping criteria 
		StatsCollector collector = new StatsCollector();
		// two-sided criteria (z score)
		double criteria = NormalCDFInverse(_probability + (1 - _probability) / 2.0);
		
		double error = Double.MAX_VALUE; 
		
		int num = 10000;
		ArrayList<float[]> priceOutput = stockPriceGeneration( num );

		// refer to the index of output of the price
		int index = 0;
		int innerindex = 0;
		
		// While error is greater than 0 and error 

		while (error > this._error || error == 0.0) {
			
			// generate the standard normal random samples.
			++ this._num_trial;
			// generate payout class
            float payout = Math.max( priceOutput.get(index)[innerindex] - (float) _strikePrice, 0 );
            ++ innerindex; 
			// adding the new data to the collector
			collector.update((double)payout);
			// compute the error (In this case, we assume that there exists cases such that error is not zero)
			error = criteria * collector.getStd() / Math.sqrt(this._num_trial);
			// Call the second time to use the antithetic paths
			if (innerindex > num - 1) {
				innerindex = 0;
				index += 1;
			}
			if (index > priceOutput.size() - 1) {
				priceOutput = stockPriceGeneration( num );
				index = 0;
				innerindex = 0;
			}
			
		}
        double price = (collector.getMean()) * Math.exp(-this._interestRate * this._length);
		StdOut.println("Final option price: " + price);
		return price;
	}
	
	/**
	 * 
	 * @param t
	 * @return the rational approximation of z-score
	 */
	double RationalApproximation( double t )
	{

	    double c[] = {2.515517, 0.802853, 0.010328};
	    double d[] = {1.432788, 0.189269, 0.001308};
	    return t - ((c[2]*t + c[1])*t + c[0]) /
	                (((d[2]*t + d[1])*t + d[0])*t + 1.0);
	}
	/**
	 * 
	 * @param p
	 * @return compute the two-sided z-score of normal distribution
	 */
	double NormalCDFInverse(double p)
	{
	    if (p <= 0.0 || p >= 1.0)
	    {
	    	throw new IllegalArgumentException("p must be between 0.0 and 1.0");
	    }
	 
	    // See article above for explanation of this section.
	    if (p < 0.5)
	    {
	        return -RationalApproximation( Math.sqrt(-2.0 * Math.log(p)) );
	    }
	    else
	    {
	        // F^-1(p) = G^-1(1-p)
	        return RationalApproximation( Math.sqrt(-2.0 * Math.log(1-p)) );
	    }
	}
	
	public static void main(String[] args) {
		
		// Define two objects
		Option<Integer, Float> IBM_eu = new Option<Integer, Float>("IBM","European",0.0001,152.35,0.01,165);
		
		// p-value, error and length
		double probability = 0.96;		
		double error = 0.1;
		float period = 252;
		IBM_eu.setDuration(period);
		
		// Simulate European option
        StdOut.println("Case 1 European Option Price:");
		Simulation_CL IBM_european = new Simulation_CL(IBM_eu, period, probability, error/2);
		IBM_european.simulate();

	}



}
