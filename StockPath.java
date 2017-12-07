package MonteCarlo;

import java.util.List;
import java.util.ArrayList;

public interface StockPath {
	
	public ArrayList<pairPrice<Integer, Double>> getPrices();
}
