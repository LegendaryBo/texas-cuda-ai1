package operatory;

import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.EvPopulation;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.general.selections.EvKBestSelection;

public class WywalDuplikaty {

  public static void aplikuj(ArrayList<EvBinaryVectorIndividual>  aIndividuals) {
    
    for (int i=0; i < aIndividuals.size(); i++) {
      
      EvBinaryVectorIndividual pIndividual = aIndividuals.get(i);
      
      for (int j=i+1 ; j < aIndividuals.size(); j++) {
        if ( pIndividual.hashCode() ==  aIndividuals.get(j).hashCode() ) {
          aIndividuals.remove(j);
          j--;
        }
      }
      
    }
    
  }
  
  
  public static void wywalUjemne(ArrayList<EvBinaryVectorIndividual>  aIndividuals) {
    
    for (int i=0; i < aIndividuals.size(); i++) {
      
      EvBinaryVectorIndividual pIndividual = aIndividuals.get(i);
      
      if (pIndividual.getObjectiveFunctionValue() < 0.0d) {
        aIndividuals.remove(i);
        i--;
      }
      
    }
    
  }
  
  public static ArrayList<EvBinaryVectorIndividual> odsiej(ArrayList<EvBinaryVectorIndividual>  aIndividuals, float ile) {
    
    EvPopulation<EvBinaryVectorIndividual> population = new EvPopulation<EvBinaryVectorIndividual>(aIndividuals);
 
    EvKBestSelection<EvBinaryVectorIndividual> selection = 
      new EvKBestSelection<EvBinaryVectorIndividual>((int)(ile*aIndividuals.size()));
    
    return selection.apply(population);
    
  }

  
  public static ArrayList<EvBinaryVectorIndividual> selekcjaDoPliku(
      ArrayList<EvBinaryVectorIndividual>  aIndividuals) {
    
    aplikuj(aIndividuals);
    
    wywalUjemne(aIndividuals);
    
    ArrayList<EvBinaryVectorIndividual>  aIndividuals2 = WywalDuplikaty.odsiej(aIndividuals, 0.5f);
    
    return aIndividuals2;
    
  }
  
  
}
