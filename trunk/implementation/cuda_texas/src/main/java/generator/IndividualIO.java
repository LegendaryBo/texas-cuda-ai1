package generator;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;

import pl.wroc.uni.ii.evolution.engine.individuals.EvBinaryVectorIndividual;

/**
 * 
 * Klasa do odczytywania/zapisywania individuali z/do pliku
 * 
 * @author Kacper Gorski
 * 
 */
public final class IndividualIO {

	/**
	 * Funkcja serializuje do wskazanego pliku osobniki.
	 * UWAGA, nie serializuje funkcji celu
	 * 
	 * @param aOsobniki
	 * @param file
	 */
	public static void zapiszDoPlikuBinarnego(ArrayList<EvBinaryVectorIndividual> aOsobniki, String file) {

		ObjectOutputStream ois = null;
		try {
			ois = new ObjectOutputStream(new FileOutputStream(file));
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {
			ois.writeInt(aOsobniki.size());
			for (int i = 0; i < aOsobniki.size(); i++) {
				int[] geny = aOsobniki.get(i).getBoolTable();
				System.out.println(geny.length);
				for (int j = 0; j < geny.length; j++) {
					ois.writeInt(geny[j]);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * metoda przepisuje n plikow z osobnikami, z formy zserialiozwanej, w forme binarna
	 * @param args - arg 1, ile osobnikow
	 */
	public static void main(String[] args) {
		int ilePlikow = 11;
		
		for (int i=1; i <= ilePlikow; i++) {
			ArrayList<EvBinaryVectorIndividual> individuale = 
				IndividualIO.odczytajZPliku( GeneratorGraczyZGeneracji.SCIEZKA + "/generacja"+ i + ".dat");
			System.out.println(GeneratorGraczyZGeneracji.SCIEZKA + "/generacja"+ i + ".bin");
			zapiszDoPlikuBinarnego(individuale, GeneratorGraczyZGeneracji.SCIEZKA + "/generacja"+ i + ".bin");
		}
		
	}
	

	/**
	 * Funkcja serializuje do wskazanego pliku osobniki, ale zapisuje tylko ich GENY, a nie caly obiekt.
	 * UWAGA, nie serializuje funkcji celu.
	 * 
	 * Format pliku:
	 * 1 int - ilosc osobnikow zapisanych w pliku
	 * reszta - ilosc intow
	 * 
	 * 
	 * @param aOsobniki
	 * @param file
	 */
	public static void zapiszDoPliku(ArrayList<EvBinaryVectorIndividual> aOsobniki, String file) {

		ObjectOutputStream ois = null;
		try {
			ois = new ObjectOutputStream(new FileOutputStream(file));
		} catch (IOException e) {
			e.printStackTrace();
		}

		try {

			for (int i = 0; i < aOsobniki.size(); i++) {
				aOsobniki.get(i).setObjectiveFunction(null);
				ois.writeObject(aOsobniki.get(i));
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public static ArrayList<EvBinaryVectorIndividual> odczytajZPliku(String file) {

		ArrayList<EvBinaryVectorIndividual> pLista = new ArrayList<EvBinaryVectorIndividual>();
		File plik = new File(file);

		BufferedInputStream stream = null;

		try {
			stream = new BufferedInputStream(new FileInputStream(plik));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		ObjectInputStream ois = null;
		try {
			ois = new ObjectInputStream(stream);
		} catch (IOException e) {
			e.printStackTrace();
		}

		while (true) {

			EvBinaryVectorIndividual individual = null;
			try {
				individual = (EvBinaryVectorIndividual) ois.readObject();
				pLista.add(individual);
			} catch (Exception e) {
				break;
			}

		}

		return pLista;
	}

}
