//
//  dataplot.h
//  gnuplot-cpp
//
//  Created by tamaki on 2013/12/26.
//  Copyright (c) 2013/12/26 tamaki. All rights reserved.
//

#ifndef __gnuplot_cpp__dataplot__
#define __gnuplot_cpp__dataplot__

#include <iostream>
/*#include <unistd.h>*/
#include <cmath>

#include <boost/assign/list_of.hpp> // for 'list_of()'

#include "gnuplot_i.hpp"


template <class T>
class multiPlotGnuplot {
protected:
    size_t length, dim;
    Gnuplot g1;
    std::vector< std::list<T> > t;
    std::vector< std::vector<T> > t_all;
    std::vector< std::string > title;
    
    
public:
    void setLength(size_t _length){
        length = _length;
    };
    void setDim(size_t _dim){
        assert(_dim > 0);
//        if(dim != _dim){
            dim = _dim;
            t.clear();
            t.resize(dim);
            t_all.clear();
            t_all.resize(dim);
            title.clear();
            title.resize(dim);
//        }
    }
    size_t getDim() const { return dim;};
    const std::vector< std::vector<T> >& getAllData() const { return &t_all; };
    
    multiPlotGnuplot(size_t _length = 20, size_t _dim = 3){
        setLength(_length);
        g1.set_style("lines");
        setDim(_dim);
    };
    ~multiPlotGnuplot(){};
    
    void
    setTitles(const std::vector< std::string > &t){
        assert ( t.size() == dim );
        std::copy (t.begin(), t.end(), title.begin() );
    }
    
    void
    plot(const std::vector< T > &d){
        assert( d.size() == title.size() );
        assert( d.size() == t.size() );
        
        
        std::vector< std::list<T> > plotdata;
        
        typename std::vector< T >::const_iterator itd = d.begin();
        typename std::vector< std::list< T > >::iterator itt = t.begin();
        typename std::vector< std::vector< T > >::iterator itt_all = t_all.begin();
        
        for(; itt != t.end(); itt++, itt_all++, itd++){
            (*itt).push_back( (*itd) );
            (*itt_all).push_back( (*itd) );
            if( (*itt).size() > length)  (*itt).pop_front();
            plotdata.push_back( (*itt) );
        }


        
        g1.reset_plot();
        g1.plot_x(plotdata, title);
        
    };
    
    friend std::ostream& operator<< (std::ostream &os, multiPlotGnuplot<T> &gp){

        for (size_t i = 0; i < gp.t_all[0].size(); i++) {
            for (size_t d = 0; d < gp.dim; d++) {
                os << gp.t_all[d][i] << " ";
            }
            os << std::endl;
        }

        
        return os;
    }
    
    bool set_GNUPlotPath(const std::string &path){
        return g1.set_GNUPlotPath(path);
    }
    
};














#endif /* defined(__gnuplot_cpp__dataplot__) */
