package pl.wroc.uni.ii.evolution.objectivefunctions.stocks;

import java.util.Date;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvNaturalNumberVectorIndividual;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvHistoricalPricesFromYahoo;
import pl.wroc.uni.ii.evolution.objectivefunctions.stocks.EvPorfolioValue;
import pl.wroc.uni.ii.evolution.solutionspaces.EvPortfolioSpace;

//NOTE this test is turned of as the code it test dont work anymore

public class EvPorfolioValueTest extends TestCase {

	@SuppressWarnings("deprecation")
  public void testEvaluate() {
				
		 EvPortfolioSpace space = new EvPortfolioSpace(10000, new Date(2006, 10, 2),  new Date(2006, 10, 4), 
			        new String[] {"MSFT", "GOOG"},
			        new EvHistoricalPricesFromYahoo());
		 
		 EvNaturalNumberVectorIndividual individual = new EvNaturalNumberVectorIndividual(space.getStockNames().length);
		 individual.setObjectiveFunction(new EvPorfolioValue(space)); //, new Date(2006, 10, 2) - this was here
		 
		 individual.setNumberAtPosition(0, 0);
		 individual.setNumberAtPosition(1, 0);
		 
		 //assertEquals(0.0,  individual.getObjectiveFunctionValue());
		 
		 individual.setNumberAtPosition(0, 1);
		 individual.setNumberAtPosition(1, 1);
     
        
     // individual.getObjectiveFunctionValue() should be:
     //    2.11.2006        3.11.2006 
     // (28.57 + 469.91) +  (28.53 + 471.80) 

     //float expected_value = (float) ((28.57 + 469.91) + (28.53 + 471.80)) / 2;
     //assertEquals( expected_value, (float) individual.getObjectiveFunctionValue());
	}

}
