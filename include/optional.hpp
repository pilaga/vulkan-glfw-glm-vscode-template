#ifndef _OPTIONAL_H_
#define _OPTIONAL_H_

// TEMPLATE CLASS:
// Class to create an object to
// store a value of any type.

// Declaration
template <class T>
class Optional {
    private:
        T element;
        bool element_exists;

    public:
        Optional();         // constructor
        ~Optional();        // destructor
        void operator=(T);  // assignment
        bool has_value();
        T value();
};

// PUBLIC methods

// constructor
template <class T>
Optional<T>::Optional() {
    element_exists = false;
}

// destructor
template <class T>
Optional<T>::~Optional() {}

// assignment operator overide
template <class T>
void Optional<T>::operator=(T val) {
    element = val;
    element_exists = true;
    return;
}

// Method to check for a valid value
template <class T>
bool Optional<T>::has_value() {
    if (element_exists) {
        return true;
    }
    return false;
}

// Method to return the value
template <class T>
T Optional<T>::value() {
    return element;
}

#endif
