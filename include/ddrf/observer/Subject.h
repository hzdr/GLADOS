#ifndef DDRF_SUBJECT_H_
#define DDRF_SUBJECT_H_

#include <list>

#include "Observer.h"

namespace ddrf
{
	class Subject
	{
		public:
			virtual ~Subject() = default;

			virtual auto attach(Observer*) -> void;
			virtual auto detach(Observer*) -> void;
			virtual auto notify() -> void;

		protected:
			Subject() = default;

		private:
			std::list<Observer*> observers_;
	};
}



#endif /* SUBJECT_H_ */
