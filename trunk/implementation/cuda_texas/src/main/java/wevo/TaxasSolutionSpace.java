package wevo;

import generator.GeneratorGraczyZGeneracji;
import generator.IndividualIO;

import java.util.ArrayList;
import java.util.Random;
import java.util.Set;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.prototype.EvObjectiveFunction;
import pl.wroc.uni.ii.evolution.engine.prototype.EvSolutionSpace;

public class TaxasSolutionSpace implements EvSolutionSpace<EvBinaryVectorIndividual> {


  public ArrayList<EvBinaryVectorIndividual> lista = null;
  private Random generatorLiczb;
  private EvObjectiveFunction<EvBinaryVectorIndividual> objFun;

  private void odczytajIndividuale(int aGeneracjaPoczatkowa, int aGeneracjaKoncowa) {
    

    lista  = new ArrayList<EvBinaryVectorIndividual>();
    for (int i=aGeneracjaPoczatkowa; i <= aGeneracjaKoncowa; i++) {

        lista.addAll( IndividualIO.odczytajZPliku(GeneratorGraczyZGeneracji.SCIEZKA+"/generacja"+i+".dat") );
    }
  }
    
    
    
  public TaxasSolutionSpace(EvObjectiveFunction<EvBinaryVectorIndividual> aObjFun, 
      int aGeneracjaPoczatkowa, int aGeneracjaKoncowa) {
    objFun = aObjFun;
    generatorLiczb = new Random();
    odczytajIndividuale(aGeneracjaPoczatkowa, aGeneracjaKoncowa);
  }
  
  
  
  /**
   * 
   */
  private static final long serialVersionUID = 1L;

  public EvBinaryVectorIndividual generateIndividual() {
    return lista.get(generatorLiczb.nextInt(lista.size()) );
  }

  public EvObjectiveFunction<EvBinaryVectorIndividual> getObjectiveFuntion() {
    return objFun;
  }
  
  
  
  
  
  
  
  
  
  
  public boolean belongsTo(EvBinaryVectorIndividual individual) {
    throw new IllegalStateException();
  }
  
  public Set divide(int n) {
    throw new IllegalStateException();
  }

  public Set divide(int n, Set p) {
      throw new IllegalStateException();
  }
  
  public void setObjectiveFuntion(EvObjectiveFunction<EvBinaryVectorIndividual> objective_function) {
    throw new IllegalStateException();
  }

  public EvBinaryVectorIndividual takeBackTo(EvBinaryVectorIndividual individual) {
    throw new IllegalStateException();
  }

}
