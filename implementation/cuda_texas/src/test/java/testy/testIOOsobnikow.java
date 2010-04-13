package testy;

import generator.IndividualIO;

import java.util.ArrayList;

import junit.framework.TestCase;
import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

public class testIOOsobnikow extends TestCase {

  public void testIO() {
    
    final String PLIK = "c:\\test_bla.dat";
    
    ArrayList<EvBinaryVectorIndividual> lista = new ArrayList<EvBinaryVectorIndividual>();
    
    EvBinaryVectorIndividual individual = new EvBinaryVectorIndividual(2);
    EvBinaryVectorIndividual individual2 = new EvBinaryVectorIndividual(13);
    individual.setGene(0, 1);
    lista.add(individual);
    lista.add(individual2);
    
    IndividualIO.zapiszDoPliku(lista, PLIK);
    
    ArrayList<EvBinaryVectorIndividual> lista2 = IndividualIO.odczytajZPliku(PLIK);
    
    EvBinaryVectorIndividual pOsobnikTestowy = lista2.get(0);
    EvBinaryVectorIndividual pOsobnikTestowy2 = lista2.get(1);
    
    assertEquals(2, pOsobnikTestowy.getDimension());
    assertEquals(13, pOsobnikTestowy2.getDimension());
  }
  
}
